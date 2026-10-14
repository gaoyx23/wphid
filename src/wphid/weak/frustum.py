"""Axisymmetric weak-library assembly on linearly mapped frustum walls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from wphid.weak.axisymmetric import axisymmetric_window_slices


@dataclass(frozen=True)
class FrustumWall:
    length: float
    inner_radius_inlet: float
    inner_radius_outlet: float
    thickness_inlet: float
    thickness_outlet: float

    def __post_init__(self):
        values = (self.length, self.inner_radius_inlet, self.inner_radius_outlet,
                  self.thickness_inlet, self.thickness_outlet)
        if not all(np.isfinite(values)) or self.length <= 0.0:
            raise ValueError("frustum dimensions must be finite and length positive")
        if min(values[1:]) <= 0.0:
            raise ValueError("frustum radii and thicknesses must be positive")

    def radius(self, z: np.ndarray, eta: np.ndarray) -> np.ndarray:
        axial_fraction = np.asarray(z, dtype=float)/self.length
        inner = self.inner_radius_inlet + axial_fraction*(
            self.inner_radius_outlet-self.inner_radius_inlet)
        thickness = self.thickness_inlet + axial_fraction*(
            self.thickness_outlet-self.thickness_inlet)
        return inner[:, None] + thickness[:, None]*np.asarray(eta, dtype=float)[None, :]

    def thickness(self, z: np.ndarray) -> np.ndarray:
        return self.thickness_inlet + np.asarray(z, dtype=float)/self.length*(
            self.thickness_outlet-self.thickness_inlet)


def frustum_volume_integral(values, t, z, eta, geometry: FrustumWall) -> float:
    time, axial, wall = _coordinates(t, z, eta, geometry)
    field = np.asarray(values, dtype=float)
    if field.shape != (time.size, axial.size, wall.size) or not np.all(np.isfinite(field)):
        raise ValueError("values must be finite with shape (n_t,n_z,n_eta)")
    radius = geometry.radius(axial, wall)
    jacobian = radius*geometry.thickness(axial)[:, None]
    return _integrate(field*jacobian[None, :, :], time, axial, wall)


def project_frustum_pde_weak(
    *, t, z, eta, geometry: FrustumWall, u, forcing, reaction_features,
    window_shape, stride_shape,
):
    """Project physical axial/radial diffusion on a mapped frustum wall.

    Physical second derivatives are transferred to the test function. The
    mapping metric includes the axial-radial cross term caused by sloping walls.
    """
    time, axial, wall = _coordinates(t, z, eta, geometry)
    shape = (time.size, axial.size, wall.size)
    values = _field(u, shape, "u")
    source = _field(forcing, shape, "forcing")
    features = np.asarray(reaction_features, dtype=float)
    if features.ndim != 4 or features.shape[:3] != shape or not np.all(np.isfinite(features)):
        raise ValueError("reaction_features must be finite with shape (n_t,n_z,n_eta,n_terms)")
    rows, targets = [], []
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        ts, zs, es = time[slices[0]], axial[slices[1]], wall[slices[2]]
        local_u, local_f, local_g = values[slices], source[slices], features[slices]
        bt, bt_t, _ = _bump(ts)
        bz, bz_z, bz_zz = _bump(zs)
        be, be_e, be_ee = _bump(es)
        radius = geometry.radius(zs, es)
        thickness = geometry.thickness(zs)
        inner_slope = (geometry.inner_radius_outlet-geometry.inner_radius_inlet)/geometry.length
        thickness_slope = (geometry.thickness_outlet-geometry.thickness_inlet)/geometry.length
        radius_z = inner_slope + es[None, :]*thickness_slope
        q = radius_z/thickness[:, None]
        q_eta = thickness_slope/thickness[:, None]
        q_z = -q*thickness_slope/thickness[:, None]

        phi = bt[:, None, None]*bz[None, :, None]*be[None, None, :]
        phi_t = bt_t[:, None, None]*bz[None, :, None]*be[None, None, :]
        phi_zz_eta = bt[:, None, None]*bz_zz[None, :, None]*be[None, None, :]
        phi_zeta = bt[:, None, None]*bz_z[None, :, None]*be_e[None, None, :]
        phi_eta = bt[:, None, None]*bz[None, :, None]*be_e[None, None, :]
        phi_etaeta = bt[:, None, None]*bz[None, :, None]*be_ee[None, None, :]
        physical_phi_zz = (phi_zz_eta - 2.0*q[None, :, :]*phi_zeta
                           + q[None, :, :]**2*phi_etaeta
                           + (q*q_eta-q_z)[None, :, :]*phi_eta)
        physical_radial = (phi_etaeta/thickness[None, :, None]**2
                           + phi_eta/(radius[None, :, :]*thickness[None, :, None]))
        axial_column = frustum_volume_integral(local_u*physical_phi_zz, ts, zs, es, geometry)
        radial_column = frustum_volume_integral(local_u*physical_radial, ts, zs, es, geometry)
        reaction = [frustum_volume_integral(local_g[..., i]*phi, ts, zs, es, geometry)
                    for i in range(features.shape[3])]
        rows.append([axial_column, radial_column, *reaction])
        targets.append(-frustum_volume_integral(local_u*phi_t, ts, zs, es, geometry)
                       - frustum_volume_integral(local_f*phi, ts, zs, es, geometry))
    return np.asarray(rows), np.asarray(targets)


def _bump(x):
    centre, half = 0.5*(x[0]+x[-1]), 0.5*(x[-1]-x[0])
    s = (x-centre)/half
    base = 1.0-s**2
    return base**4, -8.0*s*base**3/half, -8.0*base**2*(1.0-7.0*s**2)/half**2


def _integrate(values, t, z, eta):
    return float(simpson(simpson(simpson(values, x=eta, axis=2), x=z, axis=1), x=t, axis=0))


def _coordinates(t, z, eta, geometry):
    result = []
    for name, values in (("t", t), ("z", z), ("eta", eta)):
        value = np.asarray(values, dtype=float)
        if value.ndim != 1 or value.size < 3 or not np.all(np.isfinite(value)) \
                or not np.all(np.diff(value) > 0.0):
            raise ValueError(f"{name} must be finite, one-dimensional, and increasing")
        result.append(value)
    if result[1][0] < 0.0 or result[1][-1] > geometry.length:
        raise ValueError("z must lie inside the frustum length")
    if result[2][0] < 0.0 or result[2][-1] > 1.0:
        raise ValueError("eta must lie in [0,1]")
    return tuple(result)


def _field(values, shape, name):
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result

