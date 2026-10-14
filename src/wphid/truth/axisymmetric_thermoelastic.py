"""Independent local axisymmetric thermoelastic reference solver.

The production nozzle truth uses a reduced scalar stress closure.  This module
provides a deliberately separate displacement-based finite-element reference
for a thick annular wall at one axial station.  It enforces radial equilibrium
and resolves radial/hoop stress from pressure and a through-wall temperature
profile.  A declared axial-restraint fraction and the closed-end pressure
resultant are used only to assemble a three-component von Mises diagnostic.

This is a local thick-ring audit, not a full nozzle-shell or anisotropic CMC
finite-element model.  In particular it omits meridional coupling, joints,
layer architecture, geometric nonlinearities, and manufacturing residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisymmetricRingResult:
    """Stress and displacement sampled through one annular wall section."""

    sample_depth: np.ndarray
    radial_displacement: np.ndarray
    radial_stress: np.ndarray
    hoop_stress: np.ndarray
    axial_stress: np.ndarray
    von_mises_stress: np.ndarray
    mean_free_thermal_strain: float
    pressure_axial_stress: float
    element_count: int


def lame_thick_cylinder_stress(
    radii: np.ndarray,
    *,
    inner_radius: float,
    outer_radius: float,
    internal_pressure: float,
    external_pressure: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact Lamé radial and hoop stresses for a thick cylinder."""

    radii = np.asarray(radii, dtype=float)
    _validate_radii(inner_radius, outer_radius)
    if (
        radii.ndim != 1
        or radii.size == 0
        or not np.all(np.isfinite(radii))
        or np.any(radii < inner_radius)
        or np.any(radii > outer_radius)
    ):
        raise ValueError("radii must be finite points inside the annular wall")
    _validate_pressure(internal_pressure, "internal_pressure")
    _validate_pressure(external_pressure, "external_pressure")

    denominator = outer_radius**2 - inner_radius**2
    constant = (
        internal_pressure * inner_radius**2
        - external_pressure * outer_radius**2
    ) / denominator
    inverse_square = (
        inner_radius**2
        * outer_radius**2
        * (internal_pressure - external_pressure)
        / denominator
    )
    radial = constant - inverse_square / radii**2
    hoop = constant + inverse_square / radii**2
    return radial, hoop


def solve_axisymmetric_ring(
    *,
    inner_radius: float,
    outer_radius: float,
    source_depth: np.ndarray,
    temperature: np.ndarray,
    young_modulus: np.ndarray | float,
    thermal_expansion: np.ndarray | float,
    poisson_ratio: float,
    reference_temperature: float,
    internal_pressure: float,
    external_pressure: float = 0.0,
    axial_restraint_fraction: float = 0.0,
    closed_end_pressure: bool = True,
    element_count: int = 64,
    sample_depth: np.ndarray | None = None,
) -> AxisymmetricRingResult:
    """Solve a plane-stress annular section and construct a 3-D stress audit.

    ``source_depth`` and all profile inputs are measured outward from the hot
    inner surface.  Linear finite elements with two-point Gaussian quadrature
    solve radial equilibrium.  The axial stress is not part of that 2-D solve:
    it combines the exact closed-end pressure resultant with a transparent
    reduced restraint term.  Consequently the result is suitable for
    cross-checking the production closure, but not for certifying a nozzle.
    """

    _validate_radii(inner_radius, outer_radius)
    _validate_pressure(internal_pressure, "internal_pressure")
    _validate_pressure(external_pressure, "external_pressure")
    if not isinstance(element_count, (int, np.integer)) or element_count < 2:
        raise ValueError("element_count must be an integer of at least two")
    if not np.isfinite(poisson_ratio) or not 0.0 <= poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in [0, 0.5)")
    if not np.isfinite(reference_temperature) or reference_temperature <= 0.0:
        raise ValueError("reference_temperature must be positive and finite")
    if (
        not np.isfinite(axial_restraint_fraction)
        or not 0.0 <= axial_restraint_fraction <= 1.0
    ):
        raise ValueError("axial_restraint_fraction must lie in [0, 1]")

    thickness = outer_radius - inner_radius
    source_depth = _validate_depth(source_depth, thickness, "source_depth")
    temperature = _profile(temperature, source_depth, "temperature", positive=True)
    young_modulus = _profile(
        young_modulus, source_depth, "young_modulus", positive=True
    )
    thermal_expansion = _profile(
        thermal_expansion, source_depth, "thermal_expansion", positive=False
    )
    if np.any(thermal_expansion < 0.0):
        raise ValueError("thermal_expansion must be non-negative")

    if sample_depth is None:
        sample_depth = source_depth.copy()
    sample_depth = _validate_depth(sample_depth, thickness, "sample_depth")

    nodes = np.linspace(inner_radius, outer_radius, element_count + 1)
    stiffness = np.zeros((element_count + 1, element_count + 1), dtype=float)
    load = np.zeros(element_count + 1, dtype=float)
    gauss_points = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    free_strain_numerator = 0.0
    radial_measure = 0.0
    for element in range(element_count):
        element_nodes = nodes[element : element + 2]
        jacobian = 0.5 * (element_nodes[1] - element_nodes[0])
        element_stiffness = np.zeros((2, 2), dtype=float)
        element_load = np.zeros(2, dtype=float)
        for coordinate in gauss_points:
            shape = np.array(
                [0.5 * (1.0 - coordinate), 0.5 * (1.0 + coordinate)]
            )
            derivative = np.array([-0.5 / jacobian, 0.5 / jacobian])
            radius = float(shape @ element_nodes)
            depth = radius - inner_radius
            modulus = float(np.interp(depth, source_depth, young_modulus))
            expansion = float(np.interp(depth, source_depth, thermal_expansion))
            local_temperature = float(np.interp(depth, source_depth, temperature))
            constitutive = modulus / (1.0 - poisson_ratio**2) * np.array(
                [[1.0, poisson_ratio], [poisson_ratio, 1.0]], dtype=float
            )
            strain_operator = np.vstack((derivative, shape / radius))
            free_strain = expansion * (local_temperature - reference_temperature)
            weight = 2.0 * np.pi * radius * jacobian
            element_stiffness += (
                strain_operator.T @ constitutive @ strain_operator * weight
            )
            element_load += (
                strain_operator.T
                @ constitutive
                @ np.array([free_strain, free_strain])
                * weight
            )
            free_strain_numerator += free_strain * radius * jacobian
            radial_measure += radius * jacobian
        indices = np.array([element, element + 1])
        stiffness[np.ix_(indices, indices)] += element_stiffness
        load[indices] += element_load

    load[0] += internal_pressure * 2.0 * np.pi * inner_radius
    load[-1] -= external_pressure * 2.0 * np.pi * outer_radius
    displacement = np.linalg.solve(stiffness, load)

    sampled_displacement = np.empty(sample_depth.size, dtype=float)
    radial_stress = np.empty_like(sampled_displacement)
    hoop_stress = np.empty_like(sampled_displacement)
    axial_stress = np.empty_like(sampled_displacement)
    mean_free_strain = float(free_strain_numerator / radial_measure)
    pressure_axial_stress = 0.0
    if closed_end_pressure:
        pressure_axial_stress = float(
            (
                internal_pressure * inner_radius**2
                - external_pressure * outer_radius**2
            )
            / (outer_radius**2 - inner_radius**2)
        )

    for index, depth in enumerate(sample_depth):
        radius = inner_radius + float(depth)
        element = min(
            int((radius - inner_radius) / thickness * element_count),
            element_count - 1,
        )
        left, right = nodes[element : element + 2]
        element_length = right - left
        local_coordinate = 2.0 * (radius - left) / element_length - 1.0
        shape = np.array(
            [0.5 * (1.0 - local_coordinate), 0.5 * (1.0 + local_coordinate)]
        )
        derivative = np.array([-1.0 / element_length, 1.0 / element_length])
        element_displacement = displacement[element : element + 2]
        sampled_displacement[index] = float(shape @ element_displacement)
        strain = np.array(
            [
                float(derivative @ element_displacement),
                float(shape @ element_displacement) / radius,
            ]
        )
        modulus = float(np.interp(depth, source_depth, young_modulus))
        expansion = float(np.interp(depth, source_depth, thermal_expansion))
        local_temperature = float(np.interp(depth, source_depth, temperature))
        constitutive = modulus / (1.0 - poisson_ratio**2) * np.array(
            [[1.0, poisson_ratio], [poisson_ratio, 1.0]], dtype=float
        )
        stress = constitutive @ (
            strain
            - expansion
            * (local_temperature - reference_temperature)
            * np.ones(2)
        )
        radial_stress[index], hoop_stress[index] = stress
        axial_stress[index] = pressure_axial_stress - (
            axial_restraint_fraction
            * modulus
            * mean_free_strain
            / (1.0 - poisson_ratio)
        )

    von_mises = np.sqrt(
        0.5
        * (
            (radial_stress - hoop_stress) ** 2
            + (hoop_stress - axial_stress) ** 2
            + (axial_stress - radial_stress) ** 2
        )
    )
    arrays = (
        sample_depth,
        sampled_displacement,
        radial_stress,
        hoop_stress,
        axial_stress,
        von_mises,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise FloatingPointError("axisymmetric ring solve produced non-finite values")
    for values in arrays:
        values.setflags(write=False)
    return AxisymmetricRingResult(
        sample_depth=sample_depth,
        radial_displacement=sampled_displacement,
        radial_stress=radial_stress,
        hoop_stress=hoop_stress,
        axial_stress=axial_stress,
        von_mises_stress=von_mises,
        mean_free_thermal_strain=mean_free_strain,
        pressure_axial_stress=pressure_axial_stress,
        element_count=int(element_count),
    )


def _profile(
    values: np.ndarray | float,
    source_depth: np.ndarray,
    name: str,
    *,
    positive: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(source_depth.size, float(array), dtype=float)
    if array.shape != source_depth.shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and match source_depth")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _validate_depth(values: np.ndarray, thickness: float, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
        or values[0] < 0.0
        or values[-1] > thickness + max(1.0e-12, 1.0e-10 * thickness)
    ):
        raise ValueError(
            f"{name} must be a finite, strictly increasing through-wall coordinate"
        )
    return values.copy()


def _validate_radii(inner_radius: float, outer_radius: float) -> None:
    if (
        not np.isfinite(inner_radius)
        or not np.isfinite(outer_radius)
        or inner_radius <= 0.0
        or outer_radius <= inner_radius
    ):
        raise ValueError("outer_radius must exceed a positive inner_radius")


def _validate_pressure(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = (
    "AxisymmetricRingResult",
    "lame_thick_cylinder_stress",
    "solve_axisymmetric_ring",
)
