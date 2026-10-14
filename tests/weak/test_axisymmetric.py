import numpy as np
import pytest

from wphid.weak.axisymmetric import (
    axisymmetric_diffusion_weak_balance,
    axisymmetric_integral,
    axisymmetric_window_slices,
)


def _bump(x):
    coordinate = 2.0 * (x - x[0]) / (x[-1] - x[0]) - 1.0
    base = 1.0 - coordinate**2
    value = base**4
    derivative = -16.0 * coordinate * base**3 / (x[-1] - x[0])
    return value, derivative


def _manufactured(n):
    t = np.linspace(0.0, 1.0, n)
    z = np.linspace(-0.5, 0.8, n)
    r = np.linspace(0.0, 1.0, n)
    tt, zz, rr = np.meshgrid(t, z, r, indexing="ij")
    kappa = 0.17
    u = (1.0 + tt) * (zz**2 + rr**2)
    u_z = 2.0 * (1.0 + tt) * zz
    u_r = 2.0 * (1.0 + tt) * rr
    forcing = zz**2 + rr**2 - 6.0 * kappa * (1.0 + tt)
    bt, bt_t = _bump(t)
    bz, bz_z = _bump(z)
    br, br_r = _bump(r)
    phi = bt[:, None, None] * bz[None, :, None] * br[None, None, :]
    phi_t = bt_t[:, None, None] * bz[None, :, None] * br[None, None, :]
    phi_z = bt[:, None, None] * bz_z[None, :, None] * br[None, None, :]
    phi_r = bt[:, None, None] * bz[None, :, None] * br_r[None, None, :]
    return axisymmetric_diffusion_weak_balance(
        u=u, forcing=forcing, phi=phi, phi_t=phi_t, phi_z=phi_z, phi_r=phi_r,
        u_z=u_z, u_r=u_r, diffusivity=kappa, t=t, z=z, r=r,
    )


def test_axisymmetric_measure_integrates_constant_exactly_on_tensor_grid():
    t = np.linspace(0.0, 2.0, 11)
    z = np.linspace(-1.0, 3.0, 13)
    r = np.linspace(0.0, 2.0, 15)
    values = np.ones((t.size, z.size, r.size))
    # integral dt * integral dz * integral r dr = 2 * 4 * 2
    assert axisymmetric_integral(values, t, z, r) == pytest.approx(16.0)


def test_manufactured_axisymmetric_weak_residual_converges_under_refinement():
    residuals = [abs(_manufactured(n).residual) for n in (17, 33, 65)]
    assert residuals[1] < residuals[0] / 10.0
    assert residuals[2] < residuals[1] / 10.0
    assert residuals[2] < 1.0e-6


def test_axisymmetric_balance_rejects_unreported_boundary_terms():
    n = 5
    coordinate = np.linspace(0.0, 1.0, n)
    shape = (n, n, n)
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    with pytest.raises(ValueError, match="phi must vanish"):
        axisymmetric_diffusion_weak_balance(
            u=ones, forcing=zeros, phi=ones, phi_t=zeros, phi_z=zeros,
            phi_r=zeros, u_z=zeros, u_r=zeros, diffusivity=1.0,
            t=coordinate, z=coordinate, r=coordinate,
        )


def test_explicit_temporal_boundary_term_closes_noncompact_balance():
    n = 9
    t = np.linspace(0.0, 1.0, n)
    z = np.linspace(-0.5, 0.5, n)
    r = np.linspace(0.0, 1.0, n)
    tt, _, _ = np.meshgrid(t, z, r, indexing="ij")
    ones = np.ones_like(tt)
    zeros = np.zeros_like(tt)
    balance = axisymmetric_diffusion_weak_balance(
        u=tt, forcing=ones, phi=ones, phi_t=zeros, phi_z=zeros, phi_r=zeros,
        u_z=zeros, u_r=zeros, diffusivity=0.4, t=t, z=z, r=r,
        include_boundary_terms=True,
    )
    assert balance.temporal_boundary == pytest.approx(0.5)
    assert balance.spatial_boundary == pytest.approx(0.0)
    assert balance.residual == pytest.approx(0.0, abs=1e-14)


def test_explicit_axisymmetric_spatial_flux_closes_noncompact_balance():
    n = 17
    t = np.linspace(0.0, 1.0, n)
    z = np.linspace(-0.5, 0.5, n)
    r = np.linspace(0.0, 1.0, n)
    _, zz, rr = np.meshgrid(t, z, r, indexing="ij")
    kappa = 0.2
    u = zz**2 + rr**2
    ones = np.ones_like(u)
    zeros = np.zeros_like(u)
    balance = axisymmetric_diffusion_weak_balance(
        u=u, forcing=-6.0*kappa*ones, phi=ones, phi_t=zeros, phi_z=zeros,
        phi_r=zeros, u_z=2.0*zz, u_r=2.0*rr, diffusivity=kappa,
        t=t, z=z, r=r, include_boundary_terms=True,
    )
    assert balance.spatial_boundary == pytest.approx(0.6)
    assert balance.residual == pytest.approx(0.0, abs=1e-14)


def test_axisymmetric_windows_cover_only_complete_tensor_blocks():
    windows = axisymmetric_window_slices((9, 8, 7), (5, 4, 3), (4, 2, 2))
    assert len(windows) == 2 * 3 * 3
    assert windows[0] == (slice(0, 5), slice(0, 4), slice(0, 3))
    assert windows[-1] == (slice(4, 9), slice(4, 8), slice(4, 7))


@pytest.mark.parametrize("bad_radius", [np.array([-0.1, 0.0, 0.1]),
                                         np.array([0.0, 0.2, 0.1])])
def test_axisymmetric_integral_rejects_invalid_radius(bad_radius):
    coordinate = np.linspace(0.0, 1.0, 3)
    with pytest.raises(ValueError):
        axisymmetric_integral(np.ones((3, 3, 3)), coordinate, coordinate, bad_radius)
