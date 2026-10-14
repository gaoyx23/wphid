"""Local weak-library assembly for axisymmetric PDE discovery."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from wphid.weak.axisymmetric import axisymmetric_window_slices


def project_axisymmetric_pde_weak(
    *,
    t: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    u: np.ndarray,
    forcing: np.ndarray,
    reaction_features: np.ndarray,
    window_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble axial diffusion, radial diffusion, and reaction weak columns.

    For ``u_t = kz*u_zz + kr*(1/r)(r*u_r)_r + sum(c_j*g_j) + f``, this
    returns ``A, b`` with columns ``[axial, radial, g_0, ...]``. Both spatial
    derivatives are transferred to compactly supported test functions, so no
    pointwise derivative of the observed field ``u`` is computed.
    """
    time, axial, radial = (_coordinate(name, value) for name, value in
                           (("t", t), ("z", z), ("r", r)))
    shape = (time.size, axial.size, radial.size)
    values = _field(u, shape, "u")
    source = _field(forcing, shape, "forcing")
    features = np.asarray(reaction_features, dtype=float)
    if features.ndim != 4 or features.shape[:3] != shape:
        raise ValueError("reaction_features must have shape (n_t,n_z,n_r,n_terms)")
    if features.shape[3] < 1 or not np.all(np.isfinite(features)):
        raise ValueError("reaction_features must contain finite terms")

    rows, targets = [], []
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        ts, zs, rs = time[slices[0]], axial[slices[1]], radial[slices[2]]
        local_u, local_f = values[slices], source[slices]
        local_g = features[slices]
        bt, bt_t, _ = _bump_with_derivatives(ts)
        bz, _, bz_zz = _bump_with_derivatives(zs)
        br, br_r, br_rr = _bump_with_derivatives(rs)
        phi = bt[:, None, None] * bz[None, :, None] * br[None, None, :]
        phi_t = bt_t[:, None, None] * bz[None, :, None] * br[None, None, :]
        phi_zz = bt[:, None, None] * bz_zz[None, :, None] * br[None, None, :]
        axial_column = _axisymmetric_integral(local_u * phi_zz, ts, zs, rs)

        # Weighted radial self-adjoint form:
        # int u * [(r phi_r)_r] dr dz dt = int u*(r phi_rr+phi_r) dr dz dt.
        radial_weighted_test = (
            bt[:, None, None] * bz[None, :, None]
            * (rs[None, None, :] * br_rr[None, None, :] + br_r[None, None, :])
        )
        radial_column = _tensor_integral(local_u * radial_weighted_test, ts, zs, rs)
        reaction_columns = [_axisymmetric_integral(local_g[..., index] * phi, ts, zs, rs)
                            for index in range(local_g.shape[3])]
        rows.append([axial_column, radial_column, *reaction_columns])
        targets.append(-_axisymmetric_integral(local_u * phi_t, ts, zs, rs)
                       - _axisymmetric_integral(local_f * phi, ts, zs, rs))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def project_variable_axial_diffusion_probe(
    *,
    t: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    u: np.ndarray,
    coefficient: np.ndarray,
    coefficient_derivative: np.ndarray,
    window_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
) -> np.ndarray:
    """Project ``d_z(a(z) d_z u)`` as a model-adequacy probe.

    This column is not admitted to the discovered governing equation.  It is
    evaluated after discovery to test whether a low-order variable-coefficient
    operator explains structured residual left by the frozen constant-
    coefficient model.
    """
    time, axial, radial = (_coordinate(name, value) for name, value in
                           (("t", t), ("z", z), ("r", r)))
    shape = (time.size, axial.size, radial.size)
    values = _field(u, shape, "u")
    local_coefficient = np.asarray(coefficient, dtype=float)
    local_derivative = np.asarray(coefficient_derivative, dtype=float)
    if (local_coefficient.shape != axial.shape
            or local_derivative.shape != axial.shape
            or not np.all(np.isfinite(local_coefficient))
            or not np.all(np.isfinite(local_derivative))):
        raise ValueError("coefficient and derivative must be finite axial vectors")
    rows = []
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        ts, zs, rs = time[slices[0]], axial[slices[1]], radial[slices[2]]
        local_u = values[slices]
        bt, _, _ = _bump_with_derivatives(ts)
        bz, bz_z, bz_zz = _bump_with_derivatives(zs)
        br, _, _ = _bump_with_derivatives(rs)
        a = local_coefficient[slices[1]]
        a_z = local_derivative[slices[1]]
        weighted_test = (
            bt[:, None, None]
            * (a[None, :, None] * bz_zz[None, :, None]
               + a_z[None, :, None] * bz_z[None, :, None])
            * br[None, None, :]
        )
        rows.append(_axisymmetric_integral(local_u * weighted_test, ts, zs, rs))
    return np.asarray(rows, dtype=float)


def _bump_with_derivatives(x: np.ndarray):
    centre = 0.5 * (x[0] + x[-1])
    half_width = 0.5 * (x[-1] - x[0])
    coordinate = (x - centre) / half_width
    base = 1.0 - coordinate**2
    value = base**4
    first = -8.0 * coordinate * base**3 / half_width
    second = -8.0 * base**2 * (1.0 - 7.0 * coordinate**2) / half_width**2
    return value, first, second


def _tensor_integral(values, t, z, r):
    radial = simpson(values, x=r, axis=2)
    axial = simpson(radial, x=z, axis=1)
    return float(simpson(axial, x=t, axis=0))


def _axisymmetric_integral(values, t, z, r):
    """Higher-order local quadrature for derivative-bearing test functions."""
    return _tensor_integral(values * r[None, None, :], t, z, r)


def _coordinate(name, values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional coordinate")
    if not np.all(np.diff(result) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if name == "r" and result[0] < 0.0:
        raise ValueError("r cannot be negative")
    return result


def _field(values, shape, name):
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result
