import numpy as np

from wphid.weak.bell_shell import (
    bell_shell_volume_integral,
    project_bell_shell_directional_flux_pde_weak,
    project_bell_shell_flux_pde_weak,
    project_bell_shell_flux_pde_weak_with_inner_exchange,
    project_bell_shell_pde_weak,
)


def _cylindrical_shell(n=33):
    t = np.linspace(0.0, 1.2, n)
    s = np.linspace(0.0, 1.0, n)
    x = np.linspace(0.02, 0.10, n)
    z = np.broadcast_to(s[:, None], (n, n))
    r = np.broadcast_to((0.3 + x)[None, :], (n, n))
    return t, s, x, z, r


def test_bell_shell_volume_uses_axisymmetric_curved_measure():
    t, s, x, z, r = _cylindrical_shell(33)
    values = np.ones((t.size, s.size, x.size))
    expected = (t[-1] - t[0]) * (s[-1] - s[0]) * 0.5 * (
        (0.3 + x[-1]) ** 2 - (0.3 + x[0]) ** 2
    )

    result = bell_shell_volume_integral(values, t, s, x, z, r)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_bell_shell_projector_recovers_meridional_and_thickness_diffusion():
    t, s, x, z, r = _cylindrical_shell(33)
    tt, ss, xx = np.meshgrid(t, s, x, indexing="ij")
    rr = r[None, :, :]
    zeros = np.zeros_like(tt)
    cases = (
        (ss**2, -0.24 * np.ones_like(tt), 0, 0.12),
        (np.broadcast_to(rr**2, tt.shape), -0.28 * np.ones_like(tt), 1, 0.07),
    )
    for u, forcing, column, truth in cases:
        matrix, target = project_bell_shell_pde_weak(
            t=t,
            s=s,
            x=x,
            z=z,
            r=r,
            u=u,
            forcing=forcing,
            reaction_features=zeros[..., None],
            window_shape=(21, 21, 33),
            stride_shape=(4, 4, 4),
        )
        estimate = float(matrix[:, column] @ target / (matrix[:, column] @ matrix[:, column]))
        np.testing.assert_allclose(estimate, truth, rtol=0.04, atol=0.003)


def test_flux_projector_recovers_variable_conservative_diffusion():
    t, s, x, z, r = _cylindrical_shell(41)
    tt, ss, xx = np.meshgrid(t, s, x, indexing="ij")
    state = ss**2 + 0.2 * xx
    coefficient = 1.0 + 0.3 * ss
    truth = 0.08
    divergence = 2.0 + 1.2 * ss + 0.2 * coefficient / r[None, :, :]
    forcing = -truth * divergence

    matrix, target = project_bell_shell_flux_pde_weak(
        t=t,
        s=s,
        x=x,
        z=z,
        r=r,
        u=state,
        forcing=forcing,
        diffusivity_features=coefficient[..., None],
        reaction_features=np.empty((*state.shape, 0)),
        window_shape=(21, 21, 41),
        stride_shape=(10, 10, 10),
    )
    estimate = float(matrix[:, 0] @ target / (matrix[:, 0] @ matrix[:, 0]))

    np.testing.assert_allclose(estimate, truth, rtol=0.035, atol=0.002)


def test_directional_flux_projector_recovers_anisotropic_coefficients():
    t, s, x, z, r = _cylindrical_shell(41)
    tt, ss, xx = np.meshgrid(t, s, x, indexing="ij")
    rr = r[None, :, :]
    state = ss**2 * np.broadcast_to(rr**3, tt.shape)
    truth = np.array([0.12, 0.07])
    forcing = -(
        2.0 * truth[0] * rr**3 + 9.0 * truth[1] * ss**2 * rr
    ) * np.ones_like(state)
    ones = np.ones_like(state)[..., None]

    matrix, target = project_bell_shell_directional_flux_pde_weak(
        t=t,
        s=s,
        x=x,
        z=z,
        r=r,
        u=state,
        forcing=forcing,
        meridional_diffusivity_features=ones,
        thickness_diffusivity_features=ones,
        reaction_features=np.empty((*state.shape, 0)),
        window_shape=(21, 21, 41),
        stride_shape=(10, 10, 10),
    )
    estimate = np.linalg.lstsq(matrix, target, rcond=None)[0]

    np.testing.assert_allclose(estimate, truth, rtol=0.05, atol=0.005)


def test_inner_exchange_flux_projector_closes_manufactured_balance():
    t, s, x, z, r = _cylindrical_shell(41)
    tt, ss, xx = np.meshgrid(t, s, x, indexing="ij")
    state = ss**2 + 0.2 * xx
    coefficient = 1.0 + 0.3 * ss
    diffusion_truth = 0.08
    exchange_truth = 0.12
    divergence = 2.0 + 1.2 * ss + 0.2 * coefficient / r[None, :, :]
    forcing = -diffusion_truth * divergence
    outward_normal_gradient = -0.2
    exchange_feature = (
        diffusion_truth
        * coefficient[:, :, 0]
        * outward_normal_gradient
        / exchange_truth
    )

    matrix, target = project_bell_shell_flux_pde_weak_with_inner_exchange(
        t=t,
        s=s,
        x=x,
        z=z,
        r=r,
        u=state,
        forcing=forcing,
        diffusivity_features=coefficient[..., None],
        inner_exchange_features=exchange_feature[..., None],
        reaction_features=np.empty((*state.shape, 0)),
        window_shape=(21, 21, 41),
        stride_shape=(10, 10, 10),
    )
    residual = target - matrix @ np.array([diffusion_truth, exchange_truth])

    assert np.linalg.norm(residual) / np.linalg.norm(target) < 0.04


def test_outer_exchange_flux_projector_closes_manufactured_balance():
    t, s, x, z, r = _cylindrical_shell(41)
    tt, ss, xx = np.meshgrid(t, s, x, indexing="ij")
    state = ss**2 + 0.2 * xx
    coefficient = 1.0 + 0.3 * ss
    diffusion_truth = 0.08
    exchange_truth = 0.12
    divergence = 2.0 + 1.2 * ss + 0.2 * coefficient / r[None, :, :]
    forcing = -diffusion_truth * divergence
    outward_normal_gradient = 0.2
    exchange_feature = (
        diffusion_truth
        * coefficient[:, :, -1]
        * outward_normal_gradient
        / exchange_truth
    )

    matrix, target = project_bell_shell_flux_pde_weak_with_inner_exchange(
        t=t,
        s=s,
        x=x,
        z=z,
        r=r,
        u=state,
        forcing=forcing,
        diffusivity_features=coefficient[..., None],
        inner_exchange_features=exchange_feature[..., None],
        reaction_features=np.empty((*state.shape, 0)),
        window_shape=(21, 21, 41),
        stride_shape=(10, 10, 10),
        exchange_side="outer",
    )
    residual = target - matrix @ np.array([diffusion_truth, exchange_truth])

    assert np.linalg.norm(residual) / np.linalg.norm(target) < 0.04
