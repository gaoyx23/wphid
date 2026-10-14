import numpy as np

from wphid.weak.axisymmetric_library import project_axisymmetric_pde_weak


def _fields(n):
    t = np.linspace(0.0, 1.2, n)
    z = np.linspace(-0.6, 0.9, n)
    r = np.linspace(0.0, 1.0, n)
    tt, zz, rr = np.meshgrid(t, z, r, indexing="ij")
    a = np.exp(-0.4*tt)
    b = 0.7*np.exp(-0.9*tt)
    c = 0.5*np.exp(-1.3*tt)
    u = 1.0 + a*zz**2 + b*rr**2 + c*zz*rr**2
    u_t = -0.4*a*zz**2 - 0.9*b*rr**2 - 1.3*c*zz*rr**2
    u_zz = 2.0*a
    radial_laplacian = 4.0*b + 4.0*c*zz
    coefficients = np.array([0.12, 0.07, -0.25])
    forcing = u_t - coefficients[0]*u_zz - coefficients[1]*radial_laplacian \
        - coefficients[2]*u
    return t, z, r, u, forcing, coefficients


def test_closed_axisymmetric_library_recovers_manufactured_coefficients():
    t, z, r, u, forcing, truth = _fields(33)
    matrix, target = project_axisymmetric_pde_weak(
        t=t, z=z, r=r, u=u, forcing=forcing,
        reaction_features=u[..., None], window_shape=(25, 25, 25),
        stride_shape=(4, 4, 4),
    )
    estimate = np.linalg.lstsq(matrix, target, rcond=None)[0]
    np.testing.assert_allclose(estimate, truth, rtol=1.0e-2, atol=3.0e-4)
    assert matrix.shape == (27, 3)


def test_projection_uses_observed_values_without_gradient_inputs():
    t, z, r, u, forcing, _ = _fields(17)
    matrix, target = project_axisymmetric_pde_weak(
        t=t, z=z, r=r, u=u, forcing=forcing,
        reaction_features=u[..., None], window_shape=(9, 9, 9),
        stride_shape=(8, 8, 8),
    )
    assert matrix.shape == (8, 3)
    assert np.all(np.isfinite(target))
