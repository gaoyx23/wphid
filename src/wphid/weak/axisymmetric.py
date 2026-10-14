"""Axisymmetric space-time weak operators on tensor-product grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisymmetricWeakBalance:
    """Terms in the compact-support weak diffusion balance."""

    temporal: float
    diffusion: float
    forcing: float
    residual: float
    temporal_boundary: float = 0.0
    spatial_boundary: float = 0.0


def axisymmetric_integral(
    values: np.ndarray, t: np.ndarray, z: np.ndarray, r: np.ndarray,
) -> float:
    """Integrate ``values`` with the axisymmetric measure ``r dr dz dt``.

    The constant azimuthal factor ``2*pi`` is omitted because it multiplies
    every term in the weak equation and therefore cancels.
    """
    time, axial, radial = _coordinates(t, z, r)
    field = _field(values, (time.size, axial.size, radial.size), "values")
    weighted = field * radial[None, None, :]
    radial_integral = np.trapezoid(weighted, radial, axis=2)
    axial_integral = np.trapezoid(radial_integral, axial, axis=1)
    return float(np.trapezoid(axial_integral, time, axis=0))


def axisymmetric_diffusion_weak_balance(
    *,
    u: np.ndarray,
    forcing: np.ndarray,
    phi: np.ndarray,
    phi_t: np.ndarray,
    phi_z: np.ndarray,
    phi_r: np.ndarray,
    u_z: np.ndarray,
    u_r: np.ndarray,
    diffusivity: float,
    t: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    support_tolerance: float = 1.0e-12,
    include_boundary_terms: bool = False,
) -> AxisymmetricWeakBalance:
    """Evaluate the interior weak balance for ``u_t = kappa*Laplacian(u)+f``.

    The field derivatives are arguments for this P2-E0 operator-verification
    API. Later discovery stages replace them with weak library terms assembled
    from observations. ``phi`` must vanish on all six faces so no unreported
    temporal or spatial boundary contribution is silently discarded.
    """
    time, axial, radial = _coordinates(t, z, r)
    shape = (time.size, axial.size, radial.size)
    fields = {
        "u": _field(u, shape, "u"),
        "forcing": _field(forcing, shape, "forcing"),
        "phi": _field(phi, shape, "phi"),
        "phi_t": _field(phi_t, shape, "phi_t"),
        "phi_z": _field(phi_z, shape, "phi_z"),
        "phi_r": _field(phi_r, shape, "phi_r"),
        "u_z": _field(u_z, shape, "u_z"),
        "u_r": _field(u_r, shape, "u_r"),
    }
    if isinstance(diffusivity, bool):
        raise ValueError("diffusivity must be a finite non-negative number")
    kappa = float(diffusivity)
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("diffusivity must be a finite non-negative number")
    tolerance = float(support_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and non-negative")
    if not isinstance(include_boundary_terms, (bool, np.bool_)):
        raise ValueError("include_boundary_terms must be boolean")
    if not include_boundary_terms:
        _require_compact_support(fields["phi"], tolerance)

    temporal = -axisymmetric_integral(fields["u"] * fields["phi_t"], time, axial, radial)
    diffusion = kappa * axisymmetric_integral(
        fields["u_z"] * fields["phi_z"] + fields["u_r"] * fields["phi_r"],
        time, axial, radial,
    )
    forcing_term = axisymmetric_integral(
        fields["forcing"] * fields["phi"], time, axial, radial
    )
    temporal_boundary = 0.0
    spatial_boundary = 0.0
    if include_boundary_terms:
        temporal_boundary = _space_integral(
            fields["u"][-1] * fields["phi"][-1], axial, radial
        ) - _space_integral(fields["u"][0] * fields["phi"][0], axial, radial)
        # Outward normal derivatives: -u_z at z_min, +u_z at z_max,
        # -u_r at r_min, and +u_r at r_max. The r=0 contribution naturally
        # vanishes through its axisymmetric surface weight.
        axial_flux = _time_radial_integral(
            fields["phi"][:, -1, :] * fields["u_z"][:, -1, :]
            - fields["phi"][:, 0, :] * fields["u_z"][:, 0, :],
            time, radial,
        )
        radial_flux = _time_axial_integral(
            radial[-1] * fields["phi"][:, :, -1] * fields["u_r"][:, :, -1]
            - radial[0] * fields["phi"][:, :, 0] * fields["u_r"][:, :, 0],
            time, axial,
        )
        spatial_boundary = kappa * (axial_flux + radial_flux)
    return AxisymmetricWeakBalance(
        temporal=temporal,
        diffusion=diffusion,
        forcing=forcing_term,
        residual=(temporal_boundary + temporal + diffusion
                  - spatial_boundary - forcing_term),
        temporal_boundary=temporal_boundary,
        spatial_boundary=spatial_boundary,
    )


def axisymmetric_window_slices(
    shape: tuple[int, int, int],
    window_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], ...]:
    """Return deterministic complete local ``(t,z,r)`` tensor windows."""
    if len(shape) != 3 or len(window_shape) != 3 or len(stride_shape) != 3:
        raise ValueError("shape, window_shape, and stride_shape must have length 3")
    normalized = []
    for axis, (size, window, stride) in enumerate(zip(shape, window_shape, stride_shape)):
        if not all(isinstance(value, (int, np.integer)) for value in (size, window, stride)):
            raise ValueError("window dimensions must be integers")
        if size < 3 or window < 3 or window > size or stride < 1:
            raise ValueError(f"invalid window declaration on axis {axis}")
        normalized.append((int(size), int(window), int(stride)))
    starts = [range(0, size - window + 1, stride)
              for size, window, stride in normalized]
    windows = []
    for t0 in starts[0]:
        for z0 in starts[1]:
            for r0 in starts[2]:
                windows.append((slice(t0, t0 + normalized[0][1]),
                                slice(z0, z0 + normalized[1][1]),
                                slice(r0, r0 + normalized[2][1])))
    return tuple(windows)


def _space_integral(values: np.ndarray, z: np.ndarray, r: np.ndarray) -> float:
    radial_integral = np.trapezoid(values * r[None, :], r, axis=1)
    return float(np.trapezoid(radial_integral, z, axis=0))


def _time_radial_integral(values: np.ndarray, t: np.ndarray, r: np.ndarray) -> float:
    radial_integral = np.trapezoid(values * r[None, :], r, axis=1)
    return float(np.trapezoid(radial_integral, t, axis=0))


def _time_axial_integral(values: np.ndarray, t: np.ndarray, z: np.ndarray) -> float:
    axial_integral = np.trapezoid(values, z, axis=1)
    return float(np.trapezoid(axial_integral, t, axis=0))


def _coordinates(t: np.ndarray, z: np.ndarray, r: np.ndarray):
    output = []
    for name, values in (("t", t), ("z", z), ("r", r)):
        coordinate = np.asarray(values, dtype=float)
        if coordinate.ndim != 1 or coordinate.size < 3:
            raise ValueError(f"{name} must be one-dimensional with at least 3 points")
        if not np.all(np.isfinite(coordinate)) or not np.all(np.diff(coordinate) > 0.0):
            raise ValueError(f"{name} must be finite and strictly increasing")
        output.append(coordinate)
    if output[2][0] < 0.0:
        raise ValueError("r cannot contain negative radius")
    return tuple(output)


def _field(values: np.ndarray, shape: tuple[int, int, int], name: str) -> np.ndarray:
    field = np.asarray(values, dtype=float)
    if field.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must contain only finite values")
    return field


def _require_compact_support(phi: np.ndarray, tolerance: float) -> None:
    faces = (phi[0], phi[-1], phi[:, 0], phi[:, -1], phi[:, :, 0], phi[:, :, -1])
    if any(float(np.max(np.abs(face))) > tolerance for face in faces):
        raise ValueError("phi must vanish on temporal, axial, and radial boundaries")
