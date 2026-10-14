"""Auditable bridge from frozen nozzle fields to an external FEA solver.

The bridge deliberately separates three things that are easy to conflate:

1. a solver-neutral axisymmetric mesh and load contract;
2. a CalculiX input-deck adapter; and
3. a normalized result format that can be checked before comparison.

Creating a deck is not external validation.  Scientific comparison is only
authorized after a separate solver has executed the deck and its results have
passed :func:`load_external_results`.  No mechanism labels or sparse-model
coefficients are accepted by the result reader.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from wphid.truth.reusable_nozzle import NozzleGeometry


_FORBIDDEN_RESULT_TOKENS = (
    "truth",
    "ground_truth",
    "mechanism",
    "support",
    "coefficient",
    "candidate_family",
)


def _readonly(values: np.ndarray, *, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AxisymmetricQuadMesh:
    """Structured CAX4 mesh in CalculiX's radial--axial coordinate order."""

    node_ids: np.ndarray
    coordinates: np.ndarray
    parametric_coordinates: np.ndarray
    element_ids: np.ndarray
    connectivity: np.ndarray
    inner_face_elements: np.ndarray
    inlet_nodes: np.ndarray
    outlet_nodes: np.ndarray
    inner_nodes: np.ndarray
    outer_nodes: np.ndarray
    meridional_elements: int
    thickness_elements: int
    signed_areas: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.node_ids.size)

    @property
    def element_count(self) -> int:
        return int(self.element_ids.size)


@dataclass(frozen=True)
class OrthotropicMaterial:
    """Engineering constants in radial, axial and circumferential directions.

    CalculiX axisymmetric elements use global x as radial, global y as axial,
    and the expanded global z direction as circumferential.  The constants
    below follow that ordering.  Each thermal-expansion row is
    ``(temperature, alpha_radial, alpha_axial, alpha_hoop)``.
    """

    name: str
    young_radial: float
    young_axial: float
    young_hoop: float
    poisson_radial_axial: float
    poisson_radial_hoop: float
    poisson_axial_hoop: float
    shear_radial_axial: float
    shear_radial_hoop: float
    shear_axial_hoop: float
    thermal_expansion_rows: tuple[tuple[float, float, float, float], ...]
    reference_temperature: float
    density: float | None = None

    def __post_init__(self) -> None:
        elastic = np.array(
            [
                self.young_radial,
                self.young_axial,
                self.young_hoop,
                self.shear_radial_axial,
                self.shear_radial_hoop,
                self.shear_axial_hoop,
            ],
            dtype=float,
        )
        if not self.name or "," in self.name or " " in self.name:
            raise ValueError("material name must be a non-empty deck-safe token")
        if not np.all(np.isfinite(elastic)) or np.any(elastic <= 0.0):
            raise ValueError("all Young and shear moduli must be positive and finite")
        poisson = np.array(
            [
                self.poisson_radial_axial,
                self.poisson_radial_hoop,
                self.poisson_axial_hoop,
            ],
            dtype=float,
        )
        # Unlike an isotropic solid, a stable anisotropic material can have an
        # individual engineering Poisson ratio above 0.5.  Admissibility is
        # therefore decided by the positive-definite compliance check below,
        # not by the isotropic incompressibility bound.
        if not np.all(np.isfinite(poisson)):
            raise ValueError("Poisson ratios must be finite")
        if not np.isfinite(self.reference_temperature) or self.reference_temperature <= 0:
            raise ValueError("reference_temperature must be positive and finite")
        if self.density is not None and (
            not np.isfinite(self.density) or self.density <= 0.0
        ):
            raise ValueError("density must be positive and finite when supplied")
        if not self.thermal_expansion_rows:
            raise ValueError("at least one thermal-expansion row is required")
        rows = np.asarray(self.thermal_expansion_rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 4 or not np.all(np.isfinite(rows)):
            raise ValueError("thermal_expansion_rows must be finite (T, ar, aa, ah) rows")
        if np.any(rows[:, 0] <= 0.0) or np.any(np.diff(rows[:, 0]) <= 0.0):
            raise ValueError("thermal-expansion temperatures must be positive and increasing")
        if np.any(rows[:, 1:] < 0.0):
            raise ValueError("thermal-expansion coefficients must be non-negative")

        # Positive definiteness of the normal compliance is the minimum
        # admissibility check for the supplied engineering constants.
        compliance = np.array(
            [
                [1.0 / self.young_radial,
                 -self.poisson_radial_axial / self.young_radial,
                 -self.poisson_radial_hoop / self.young_radial],
                [-self.poisson_radial_axial / self.young_radial,
                 1.0 / self.young_axial,
                 -self.poisson_axial_hoop / self.young_axial],
                [-self.poisson_radial_hoop / self.young_radial,
                 -self.poisson_axial_hoop / self.young_axial,
                 1.0 / self.young_hoop],
            ],
            dtype=float,
        )
        if np.min(np.linalg.eigvalsh(compliance)) <= 0.0:
            raise ValueError("engineering constants do not define a stable compliance")

    @classmethod
    def isotropic_equivalent(
        cls,
        *,
        name: str,
        young_modulus: float,
        poisson_ratio: float,
        thermal_expansion_rows: Iterable[tuple[float, float]],
        reference_temperature: float,
        density: float | None = None,
    ) -> "OrthotropicMaterial":
        """Represent an isotropic material through the orthotropic interface."""

        shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
        rows = tuple((temperature, alpha, alpha, alpha) for temperature, alpha in thermal_expansion_rows)
        return cls(
            name=name,
            young_radial=young_modulus,
            young_axial=young_modulus,
            young_hoop=young_modulus,
            poisson_radial_axial=poisson_ratio,
            poisson_radial_hoop=poisson_ratio,
            poisson_axial_hoop=poisson_ratio,
            shear_radial_axial=shear,
            shear_radial_hoop=shear,
            shear_axial_hoop=shear,
            thermal_expansion_rows=rows,
            reference_temperature=reference_temperature,
            density=density,
        )


@dataclass(frozen=True)
class ExternalFEAResults:
    """Normalized external nodal and element-centred result arrays."""

    node_ids: np.ndarray
    radial_displacement: np.ndarray
    axial_displacement: np.ndarray
    temperature: np.ndarray
    element_ids: np.ndarray
    radial_stress: np.ndarray
    axial_stress: np.ndarray
    hoop_stress: np.ndarray
    radial_axial_shear: np.ndarray
    von_mises_stress: np.ndarray


def _assemble_mesh(
    radial: np.ndarray,
    axial: np.ndarray,
) -> AxisymmetricQuadMesh:
    radial = np.asarray(radial, dtype=float)
    axial = np.asarray(axial, dtype=float)
    if radial.shape != axial.shape or radial.ndim != 2:
        raise ValueError("radial and axial node arrays must be matching 2-D arrays")
    if min(radial.shape) < 2 or not np.all(np.isfinite(radial)) or not np.all(np.isfinite(axial)):
        raise ValueError("mesh coordinates must be finite and contain at least one element")
    if np.any(radial <= 0.0):
        raise ValueError("axisymmetric radial coordinates must be strictly positive")

    meridional_elements = radial.shape[0] - 1
    thickness_elements = radial.shape[1] - 1
    node_ids = np.arange(1, radial.size + 1, dtype=int)
    coordinates = np.column_stack((radial.ravel(), axial.ravel(), np.zeros(radial.size)))
    s_fraction = np.repeat(
        np.linspace(0.0, 1.0, meridional_elements + 1), thickness_elements + 1
    )
    x_fraction = np.tile(
        np.linspace(0.0, 1.0, thickness_elements + 1), meridional_elements + 1
    )
    parametric = np.column_stack((s_fraction, x_fraction))

    def node(i: int, j: int) -> int:
        return i * (thickness_elements + 1) + j + 1

    connectivity: list[list[int]] = []
    inner_faces: list[int] = []
    for i in range(meridional_elements):
        for j in range(thickness_elements):
            # Counter-clockwise order in the radial--axial plane.  Face 4
            # (node 4 to node 1) is therefore the gas-side inner wall.
            connectivity.append(
                [node(i, j), node(i, j + 1), node(i + 1, j + 1), node(i + 1, j)]
            )
            if j == 0:
                inner_faces.append(i * thickness_elements + j + 1)
    connectivity_array = np.asarray(connectivity, dtype=int)
    element_ids = np.arange(1, connectivity_array.shape[0] + 1, dtype=int)

    points = coordinates[connectivity_array - 1, :2]
    next_points = np.roll(points, shift=-1, axis=1)
    signed_areas = 0.5 * np.sum(
        points[:, :, 0] * next_points[:, :, 1]
        - next_points[:, :, 0] * points[:, :, 1],
        axis=1,
    )
    if not np.all(signed_areas > 0.0):
        raise ValueError("axisymmetric mesh contains inverted or degenerate elements")

    inlet_nodes = np.arange(1, thickness_elements + 2, dtype=int)
    outlet_start = meridional_elements * (thickness_elements + 1) + 1
    outlet_nodes = np.arange(outlet_start, outlet_start + thickness_elements + 1, dtype=int)
    inner_nodes = np.arange(
        1,
        (meridional_elements + 1) * (thickness_elements + 1) + 1,
        thickness_elements + 1,
        dtype=int,
    )
    outer_nodes = inner_nodes + thickness_elements
    return AxisymmetricQuadMesh(
        node_ids=_readonly(node_ids, dtype=int),
        coordinates=_readonly(coordinates),
        parametric_coordinates=_readonly(parametric),
        element_ids=_readonly(element_ids, dtype=int),
        connectivity=_readonly(connectivity_array, dtype=int),
        inner_face_elements=_readonly(np.asarray(inner_faces), dtype=int),
        inlet_nodes=_readonly(inlet_nodes, dtype=int),
        outlet_nodes=_readonly(outlet_nodes, dtype=int),
        inner_nodes=_readonly(inner_nodes, dtype=int),
        outer_nodes=_readonly(outer_nodes, dtype=int),
        meridional_elements=meridional_elements,
        thickness_elements=thickness_elements,
        signed_areas=_readonly(signed_areas),
    )


def build_nozzle_axisymmetric_mesh(
    geometry: NozzleGeometry | None = None,
    *,
    meridional_elements: int = 48,
    thickness_elements: int = 8,
) -> AxisymmetricQuadMesh:
    """Create a body-fitted mesh of the actual smooth bell-wall envelope."""

    geometry = NozzleGeometry() if geometry is None else geometry
    if not isinstance(meridional_elements, int) or meridional_elements < 2:
        raise ValueError("meridional_elements must be an integer of at least two")
    if not isinstance(thickness_elements, int) or thickness_elements < 1:
        raise ValueError("thickness_elements must be a positive integer")
    base_axial = np.linspace(0.0, geometry.length, meridional_elements + 1)
    inner_radius = geometry.inner_radius(base_axial)
    slope = geometry.inner_slope(base_axial)
    normal_radial = 1.0 / np.sqrt(1.0 + slope**2)
    normal_axial = -slope * normal_radial
    depth = np.linspace(0.0, geometry.wall_thickness, thickness_elements + 1)
    radial = inner_radius[:, None] + normal_radial[:, None] * depth[None, :]
    axial = base_axial[:, None] + normal_axial[:, None] * depth[None, :]
    return _assemble_mesh(radial, axial)


def build_axisymmetric_strip_mesh(
    *,
    inner_radius: float,
    wall_thickness: float,
    axial_length: float,
    axial_elements: int = 12,
    thickness_elements: int = 8,
) -> AxisymmetricQuadMesh:
    """Create a straight annular strip for Lamé and free-heating checks."""

    scalars = np.asarray([inner_radius, wall_thickness, axial_length], dtype=float)
    if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
        raise ValueError("strip dimensions must be positive and finite")
    if not isinstance(axial_elements, int) or axial_elements < 1:
        raise ValueError("axial_elements must be a positive integer")
    if not isinstance(thickness_elements, int) or thickness_elements < 1:
        raise ValueError("thickness_elements must be a positive integer")
    radial_line = np.linspace(
        inner_radius, inner_radius + wall_thickness, thickness_elements + 1
    )
    axial_line = np.linspace(0.0, axial_length, axial_elements + 1)
    radial = np.broadcast_to(radial_line, (axial_elements + 1, radial_line.size))
    axial = np.broadcast_to(axial_line[:, None], radial.shape)
    return _assemble_mesh(radial, axial)


def interpolate_cell_field_to_nodes(
    cell_field: np.ndarray,
    mesh: AxisymmetricQuadMesh,
) -> np.ndarray:
    """Map a structured cell-centred field to independent mesh nodes.

    Parametric coordinates, rather than copied physical coordinates, are used
    because both truth and external meshes are body-fitted to the same wall but
    deliberately use different resolutions.  Boundary values are extended from
    the nearest cell centre instead of being extrapolated.
    """

    values = np.asarray(cell_field, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("cell_field must be a non-empty 2-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("cell_field must contain only finite values")
    s_centres = (np.arange(values.shape[0], dtype=float) + 0.5) / values.shape[0]
    x_centres = (np.arange(values.shape[1], dtype=float) + 0.5) / values.shape[1]
    s_axis = np.concatenate(([0.0], s_centres, [1.0]))
    x_axis = np.concatenate(([0.0], x_centres, [1.0]))
    extended = np.pad(values, ((1, 1), (1, 1)), mode="edge")
    interpolator = RegularGridInterpolator(
        (s_axis, x_axis), extended, bounds_error=True
    )
    mapped = interpolator(mesh.parametric_coordinates)
    return _readonly(mapped)


def interpolate_cell_profile_to_inner_faces(
    cell_profile: np.ndarray,
    mesh: AxisymmetricQuadMesh,
) -> np.ndarray:
    """Map a cell-centred axial profile to gas-side element-face centres."""

    values = np.asarray(cell_profile, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("cell_profile must be a non-empty finite 1-D array")
    source = (np.arange(values.size, dtype=float) + 0.5) / values.size
    source = np.concatenate(([0.0], source, [1.0]))
    extended = np.concatenate(([values[0]], values, [values[-1]]))
    target = (
        np.arange(mesh.meridional_elements, dtype=float) + 0.5
    ) / mesh.meridional_elements
    return _readonly(np.interp(target, source, extended))


def _set_lines(name: str, identifiers: np.ndarray, *, per_line: int = 16) -> list[str]:
    lines = [f"*NSET,NSET={name}"]
    ids = [str(int(value)) for value in identifiers]
    lines.extend(",".join(ids[index : index + per_line]) for index in range(0, len(ids), per_line))
    return lines


def render_calculix_static_deck(
    mesh: AxisymmetricQuadMesh,
    *,
    nodal_temperature: np.ndarray,
    inner_pressure: np.ndarray,
    material: OrthotropicMaterial,
    title: str,
    inlet_boundary: str = "axial_roller",
    inlet_axial_foundation_stiffness_per_area: float | None = None,
) -> str:
    """Render a pure structural CAX4 thermoelastic CalculiX input deck."""

    temperature = np.asarray(nodal_temperature, dtype=float)
    pressure = np.asarray(inner_pressure, dtype=float)
    if temperature.shape != (mesh.node_count,) or not np.all(np.isfinite(temperature)):
        raise ValueError("nodal_temperature must be finite with one value per node")
    if np.any(temperature <= 0.0):
        raise ValueError("nodal temperatures must be positive")
    if pressure.shape != (mesh.meridional_elements,) or not np.all(np.isfinite(pressure)):
        raise ValueError("inner_pressure must be finite with one value per inner face")
    if np.any(pressure < 0.0):
        raise ValueError("inner pressure cannot be negative")
    if inlet_boundary not in {
        "axial_roller",
        "single_node_axial",
        "all_axial_restrained",
        "inlet_clamped",
        "axial_foundation",
    }:
        raise ValueError("unsupported inlet_boundary")
    if inlet_boundary == "axial_foundation":
        if (
            inlet_axial_foundation_stiffness_per_area is None
            or not np.isfinite(inlet_axial_foundation_stiffness_per_area)
            or inlet_axial_foundation_stiffness_per_area <= 0.0
        ):
            raise ValueError(
                "axial_foundation requires a positive finite stiffness per area"
            )
    elif inlet_axial_foundation_stiffness_per_area is not None:
        raise ValueError("foundation stiffness is only valid for axial_foundation")
    if any(token in title.lower() for token in _FORBIDDEN_RESULT_TOKENS):
        raise ValueError("deck title contains a prohibited information-leakage token")

    lines = [
        "** WPHID EXTERNAL FEA CONTRACT 1.0",
        f"** {title}",
        "** Coordinates: x=radial, y=axial, z=0; SI units",
        "** CAX4 face 4 is the gas-side inner wall",
        "*NODE,NSET=NALL",
    ]
    for node_id, coordinate in zip(mesh.node_ids, mesh.coordinates, strict=True):
        lines.append(
            f"{int(node_id)},{coordinate[0]:.12e},{coordinate[1]:.12e},0.0"
        )
    lines.append("*ELEMENT,TYPE=CAX4,ELSET=EALL")
    for element_id, nodes in zip(mesh.element_ids, mesh.connectivity, strict=True):
        lines.append(f"{int(element_id)}," + ",".join(str(int(node)) for node in nodes))
    lines.extend(_set_lines("INLET", mesh.inlet_nodes))
    lines.extend(_set_lines("OUTLET", mesh.outlet_nodes))
    lines.extend(_set_lines("INNER", mesh.inner_nodes))
    lines.extend(_set_lines("OUTER", mesh.outer_nodes))
    if inlet_boundary == "axial_foundation":
        inlet_radii = mesh.coordinates[mesh.inlet_nodes - 1, 0]
        radial_edges = np.concatenate(
            (
                inlet_radii[:1],
                0.5 * (inlet_radii[:-1] + inlet_radii[1:]),
                inlet_radii[-1:],
            )
        )
        nodal_annular_areas = np.pi * (
            radial_edges[1:] ** 2 - radial_edges[:-1] ** 2
        )
        for offset, (node_id, nodal_area) in enumerate(
            zip(mesh.inlet_nodes, nodal_annular_areas, strict=True), start=1
        ):
            spring_id = mesh.element_count + offset
            spring_set = f"IAXSPR{offset}"
            spring_stiffness = (
                float(inlet_axial_foundation_stiffness_per_area) * nodal_area
            )
            lines.extend(
                [
                    f"*ELEMENT,TYPE=SPRING1,ELSET={spring_set}",
                    f"{spring_id},{int(node_id)}",
                    f"*SPRING,ELSET={spring_set}",
                    "2",
                    f"{spring_stiffness:.12e}",
                ]
            )
    lines.extend(
        [
            f"*MATERIAL,NAME={material.name}",
            "*ELASTIC,TYPE=ENGINEERING CONSTANTS",
            (
                f"{material.young_radial:.12e},{material.young_axial:.12e},"
                f"{material.young_hoop:.12e},{material.poisson_radial_axial:.12e},"
                f"{material.poisson_radial_hoop:.12e},{material.poisson_axial_hoop:.12e},"
                f"{material.shear_radial_axial:.12e},{material.shear_radial_hoop:.12e}"
            ),
            f"{material.shear_axial_hoop:.12e},{material.reference_temperature:.12e}",
            (
                f"*EXPANSION,TYPE=ORTHO,ZERO={material.reference_temperature:.12e}"
            ),
        ]
    )
    for row in material.thermal_expansion_rows:
        local_temperature, radial_alpha, axial_alpha, hoop_alpha = row
        lines.append(
            f"{radial_alpha:.12e},{axial_alpha:.12e},{hoop_alpha:.12e},"
            f"{local_temperature:.12e}"
        )
    if material.density is not None:
        lines.extend(["*DENSITY", f"{material.density:.12e}"])
    lines.extend(
        [
            f"*SOLID SECTION,ELSET=EALL,MATERIAL={material.name}",
            "*INITIAL CONDITIONS,TYPE=TEMPERATURE",
            f"NALL,{material.reference_temperature:.12e}",
            "*STEP,NLGEOM=NO",
            "*STATIC,DIRECT",
            "1.0,1.0",
        ]
    )
    if inlet_boundary == "axial_roller":
        lines.append("*BOUNDARY")
        lines.append("INLET,2,2,0.0")
    elif inlet_boundary == "single_node_axial":
        lines.append("*BOUNDARY")
        lines.append(f"{int(mesh.inlet_nodes[0])},2,2,0.0")
    elif inlet_boundary == "all_axial_restrained":
        lines.append("*BOUNDARY")
        lines.append("NALL,2,2,0.0")
    elif inlet_boundary == "inlet_clamped":
        lines.append("*BOUNDARY")
        lines.append("INLET,1,2,0.0")
    lines.append("*TEMPERATURE,OP=NEW")
    for node_id, value in zip(mesh.node_ids, temperature, strict=True):
        lines.append(f"{int(node_id)},{value:.12e}")
    lines.append("*DLOAD,OP=NEW")
    for element_id, value in zip(mesh.inner_face_elements, pressure, strict=True):
        lines.append(f"{int(element_id)},P4,{value:.12e}")
    lines.extend(
        [
            "*NODE FILE",
            "U,NT",
            "*EL FILE",
            "S,E",
            "*END STEP",
        ]
    )
    return "\n".join(lines) + "\n"


def _keyword_blocks(deck: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_keyword: str | None = None
    current_rows: list[str] = []
    for raw_line in deck.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            if current_keyword is not None:
                blocks.append((current_keyword, current_rows))
            current_keyword = line
            current_rows = []
        elif current_keyword is not None:
            current_rows.append(line)
    if current_keyword is not None:
        blocks.append((current_keyword, current_rows))
    return blocks


def audit_calculix_deck(
    deck: str,
    mesh: AxisymmetricQuadMesh,
) -> dict[str, int | float | bool | str]:
    """Perform a strict structural and leakage preflight on a rendered deck."""

    blocks = _keyword_blocks(deck)
    by_keyword: dict[str, list[list[str]]] = {}
    for keyword, rows in blocks:
        base = keyword.split(",", maxsplit=1)[0].upper()
        by_keyword.setdefault(base, []).append(rows)
    required = {
        "*NODE",
        "*ELEMENT",
        "*MATERIAL",
        "*ELASTIC",
        "*EXPANSION",
        "*SOLID SECTION",
        "*STEP",
        "*STATIC",
        "*TEMPERATURE",
        "*DLOAD",
        "*NODE FILE",
        "*EL FILE",
        "*END STEP",
    }
    required_present = required.issubset(by_keyword)
    has_admissible_support = "*BOUNDARY" in by_keyword or "*SPRING" in by_keyword
    node_rows = by_keyword.get("*NODE", [[]])[0]
    element_rows = by_keyword.get("*ELEMENT", [[]])[0]
    temperature_rows = by_keyword.get("*TEMPERATURE", [[]])[0]
    pressure_rows = by_keyword.get("*DLOAD", [[]])[0]

    parsed_node_ids = [int(row.split(",")[0]) for row in node_rows]
    parsed_element_ids = [int(row.split(",")[0]) for row in element_rows]
    loaded_elements: list[int] = []
    all_pressure_faces_are_inner_face_four = True
    pressure_values: list[float] = []
    for row in pressure_rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3 or parts[1].upper() != "P4":
            all_pressure_faces_are_inner_face_four = False
            continue
        loaded_elements.append(int(parts[0]))
        pressure_values.append(float(parts[2]))
    temperature_values = [float(row.split(",")[1]) for row in temperature_rows]
    leakage_free = not any(token in deck.lower() for token in _FORBIDDEN_RESULT_TOKENS)
    all_inner_faces_loaded_once = (
        len(loaded_elements) == len(set(loaded_elements))
        and set(loaded_elements) == set(mesh.inner_face_elements.tolist())
    )
    passed = bool(
        required_present
        and has_admissible_support
        and parsed_node_ids == mesh.node_ids.tolist()
        and parsed_element_ids == mesh.element_ids.tolist()
        and len(temperature_rows) == mesh.node_count
        and all_pressure_faces_are_inner_face_four
        and all_inner_faces_loaded_once
        and temperature_values
        and min(temperature_values) > 0.0
        and (not pressure_values or min(pressure_values) >= 0.0)
        and leakage_free
        and np.all(mesh.signed_areas > 0.0)
    )
    return {
        "passed": passed,
        "required_keywords_present": required_present,
        "admissible_support_present": has_admissible_support,
        "support_representation": (
            "distributed_spring" if "*SPRING" in by_keyword else "kinematic_boundary"
        ),
        "node_count": len(node_rows),
        "element_count": len(element_rows),
        "temperature_count": len(temperature_rows),
        "pressure_face_count": len(pressure_rows),
        "all_elements_positive_area": bool(np.all(mesh.signed_areas > 0.0)),
        "minimum_signed_area_m2": float(np.min(mesh.signed_areas)),
        "all_pressure_faces_are_p4": all_pressure_faces_are_inner_face_four,
        "all_inner_faces_loaded_once": all_inner_faces_loaded_once,
        "information_leakage_tokens_absent": leakage_free,
        "coordinate_convention": "x=radial,y=axial,z=0",
    }


def external_result_schema() -> Mapping[str, tuple[str, ...]]:
    """Return the normalized CSV columns expected from any external solver."""

    return {
        "nodal_results.csv": (
            "node_id",
            "r_m",
            "z_m",
            "u_radial_m",
            "u_axial_m",
            "temperature_k",
        ),
        "element_results.csv": (
            "element_id",
            "sigma_radial_pa",
            "sigma_axial_pa",
            "sigma_hoop_pa",
            "tau_radial_axial_pa",
            "von_mises_pa",
        ),
    }


def _read_numeric_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        lowered = tuple(name.strip().lower() for name in reader.fieldnames)
        if any(token in name for name in lowered for token in _FORBIDDEN_RESULT_TOKENS):
            raise ValueError(f"{path.name} contains a prohibited information-leakage column")
        if set(lowered) != set(required_columns):
            raise ValueError(
                f"{path.name} columns must exactly match the normalized result contract"
            )
        rows: list[dict[str, float]] = []
        for source_row in reader:
            normalized = {key.strip().lower(): value for key, value in source_row.items()}
            try:
                row = {name: float(normalized[name]) for name in required_columns}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name} contains a non-numeric result") from exc
            if not np.all(np.isfinite(list(row.values()))):
                raise ValueError(f"{path.name} contains a non-finite result")
            rows.append(row)
    return rows


def load_external_results(
    nodal_results_csv: str | Path,
    element_results_csv: str | Path,
    mesh: AxisymmetricQuadMesh,
    *,
    coordinate_tolerance: float = 1.0e-8,
    von_mises_relative_tolerance: float = 1.0e-5,
) -> ExternalFEAResults:
    """Read, align and validate normalized results from an independent solver."""

    if coordinate_tolerance <= 0.0 or von_mises_relative_tolerance <= 0.0:
        raise ValueError("result tolerances must be positive")
    schema = external_result_schema()
    nodal_rows = _read_numeric_csv(Path(nodal_results_csv), schema["nodal_results.csv"])
    element_rows = _read_numeric_csv(
        Path(element_results_csv), schema["element_results.csv"]
    )
    if len(nodal_rows) != mesh.node_count or len(element_rows) != mesh.element_count:
        raise ValueError("external result row counts do not match the contract mesh")

    nodal_by_id = {int(row["node_id"]): row for row in nodal_rows}
    element_by_id = {int(row["element_id"]): row for row in element_rows}
    if len(nodal_by_id) != len(nodal_rows) or set(nodal_by_id) != set(mesh.node_ids.tolist()):
        raise ValueError("external nodal IDs are duplicate or do not match the contract")
    if len(element_by_id) != len(element_rows) or set(element_by_id) != set(mesh.element_ids.tolist()):
        raise ValueError("external element IDs are duplicate or do not match the contract")

    ordered_nodes = [nodal_by_id[int(node_id)] for node_id in mesh.node_ids]
    returned_coordinates = np.array(
        [[row["r_m"], row["z_m"]] for row in ordered_nodes], dtype=float
    )
    maximum_coordinate_error = float(
        np.max(np.abs(returned_coordinates - mesh.coordinates[:, :2]))
    )
    if maximum_coordinate_error > coordinate_tolerance:
        raise ValueError("external nodal coordinates do not match the contract mesh")

    ordered_elements = [element_by_id[int(element_id)] for element_id in mesh.element_ids]
    radial_stress = np.array([row["sigma_radial_pa"] for row in ordered_elements])
    axial_stress = np.array([row["sigma_axial_pa"] for row in ordered_elements])
    hoop_stress = np.array([row["sigma_hoop_pa"] for row in ordered_elements])
    shear_stress = np.array([row["tau_radial_axial_pa"] for row in ordered_elements])
    reported_von_mises = np.array([row["von_mises_pa"] for row in ordered_elements])
    reconstructed_von_mises = np.sqrt(
        0.5
        * (
            (radial_stress - axial_stress) ** 2
            + (axial_stress - hoop_stress) ** 2
            + (hoop_stress - radial_stress) ** 2
        )
        + 3.0 * shear_stress**2
    )
    denominator = np.maximum(np.abs(reconstructed_von_mises), 1.0)
    maximum_relative_error = float(
        np.max(np.abs(reported_von_mises - reconstructed_von_mises) / denominator)
    )
    if maximum_relative_error > von_mises_relative_tolerance:
        raise ValueError("reported von Mises stress is inconsistent with tensor components")

    return ExternalFEAResults(
        node_ids=_readonly(mesh.node_ids, dtype=int),
        radial_displacement=_readonly(
            np.array([row["u_radial_m"] for row in ordered_nodes])
        ),
        axial_displacement=_readonly(
            np.array([row["u_axial_m"] for row in ordered_nodes])
        ),
        temperature=_readonly(np.array([row["temperature_k"] for row in ordered_nodes])),
        element_ids=_readonly(mesh.element_ids, dtype=int),
        radial_stress=_readonly(radial_stress),
        axial_stress=_readonly(axial_stress),
        hoop_stress=_readonly(hoop_stress),
        radial_axial_shear=_readonly(shear_stress),
        von_mises_stress=_readonly(reported_von_mises),
    )


# CalculiX ASCII FRD uses a three-digit exponent without separators between
# adjacent values.  Capping the exponent width is essential: an unrestricted
# ``\d+`` would consume the first digit of the next positive component.
_FRD_FLOAT = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+)E[+-]\d{2,3}")


def _read_calculix_frd_fields(
    path: Path,
) -> tuple[str, dict[str, dict[int, np.ndarray]]]:
    """Read the nodal DISP, NDTEMP and STRESS blocks from ASCII FRD output."""

    version = "unknown"
    active_label: str | None = None
    fields: dict[str, dict[int, np.ndarray]] = {}
    with path.open("r", encoding="ascii", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith("    1UVERSION"):
                version = line[len("    1UVERSION") :].strip()
            if line.startswith(" -4"):
                parts = line.split()
                active_label = parts[1].upper() if len(parts) >= 2 else None
                if active_label is not None:
                    fields[active_label] = {}
                continue
            if line.startswith(" -3"):
                active_label = None
                continue
            if active_label is None or not line.startswith(" -1"):
                continue
            try:
                node_id = int(line[3:13])
            except ValueError as exc:
                raise ValueError(f"invalid FRD node field in {path.name}") from exc
            values = np.array(
                [float(value) for value in _FRD_FLOAT.findall(line[13:])],
                dtype=float,
            )
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid FRD result row in {path.name}")
            fields[active_label][node_id] = values
    return version, fields


def convert_calculix_frd_to_normalized_csv(
    frd_path: str | Path,
    mesh: AxisymmetricQuadMesh,
    nodal_results_csv: str | Path,
    element_results_csv: str | Path,
    *,
    allow_additional_nodes: bool = False,
) -> dict[str, str | int | float]:
    """Convert CalculiX ASCII FRD output into the strict solver-neutral schema.

    CalculiX writes extrapolated nodal stresses for CAX4 elements.  The
    normalized element record is the arithmetic mean of the four connected
    nodal tensors; von Mises stress is then recomputed from that mean tensor.
    The normalized files can subsequently be checked with
    :func:`load_external_results`.
    """

    frd_path = Path(frd_path)
    version, fields = _read_calculix_frd_fields(frd_path)
    required = {"DISP": 2, "NDTEMP": 1, "STRESS": 4}
    for label, minimum_components in required.items():
        if label not in fields:
            raise ValueError(f"{frd_path.name} is missing the {label} field")
        returned_ids = set(fields[label])
        expected_ids = set(mesh.node_ids.tolist())
        ids_match = (
            expected_ids.issubset(returned_ids)
            if allow_additional_nodes
            else returned_ids == expected_ids
        )
        if not ids_match:
            raise ValueError(f"{frd_path.name} {label} node IDs do not match the mesh")
        if min(values.size for values in fields[label].values()) < minimum_components:
            raise ValueError(f"{frd_path.name} {label} has too few components")

    displacement = np.vstack([fields["DISP"][int(node)] for node in mesh.node_ids])
    temperature = np.array(
        [fields["NDTEMP"][int(node)][0] for node in mesh.node_ids], dtype=float
    )
    nodal_stress = np.vstack(
        [fields["STRESS"][int(node)][:6] for node in mesh.node_ids]
    )
    if nodal_stress.shape[1] < 6:
        raise ValueError(f"{frd_path.name} STRESS must contain six tensor components")

    nodal_results_csv = Path(nodal_results_csv)
    with nodal_results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(external_result_schema()["nodal_results.csv"])
        for index, node_id in enumerate(mesh.node_ids):
            writer.writerow(
                [
                    int(node_id),
                    float(mesh.coordinates[index, 0]),
                    float(mesh.coordinates[index, 1]),
                    float(displacement[index, 0]),
                    float(displacement[index, 1]),
                    float(temperature[index]),
                ]
            )

    # FRD ordering is SXX, SYY, SZZ, SXY, SYZ, SZX.  For CAX4 these are
    # radial, axial, circumferential, radial-axial, axial-hoop and hoop-radial.
    element_stress = np.mean(nodal_stress[mesh.connectivity - 1], axis=1)
    radial = element_stress[:, 0]
    axial = element_stress[:, 1]
    hoop = element_stress[:, 2]
    radial_axial = element_stress[:, 3]
    von_mises = np.sqrt(
        0.5
        * ((radial - axial) ** 2 + (axial - hoop) ** 2 + (hoop - radial) ** 2)
        + 3.0
        * (
            radial_axial**2
            + element_stress[:, 4] ** 2
            + element_stress[:, 5] ** 2
        )
    )
    element_results_csv = Path(element_results_csv)
    with element_results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(external_result_schema()["element_results.csv"])
        for index, element_id in enumerate(mesh.element_ids):
            writer.writerow(
                [
                    int(element_id),
                    float(radial[index]),
                    float(axial[index]),
                    float(hoop[index]),
                    float(radial_axial[index]),
                    float(von_mises[index]),
                ]
            )
    return {
        "solver_version": version,
        "frd_node_count": len(fields["STRESS"]),
        "normalized_node_count": mesh.node_count,
        "additional_frd_nodes_ignored": len(fields["STRESS"]) - mesh.node_count,
        "normalized_element_count": mesh.element_count,
        "maximum_abs_out_of_plane_shear_pa": float(
            np.max(np.abs(element_stress[:, 4:6]))
        ),
        "stress_location_conversion": "FRD extrapolated nodes to four-node element mean",
    }
