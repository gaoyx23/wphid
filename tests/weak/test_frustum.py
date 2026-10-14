import numpy as np

from wphid.weak.frustum import FrustumWall, frustum_volume_integral, project_frustum_pde_weak


def test_frustum_volume_matches_analytic_axisymmetric_measure():
    geometry = FrustumWall(1.2, 0.3, 0.6, 0.08, 0.08)
    t = np.linspace(0.0, 2.0, 17)
    z = np.linspace(0.0, geometry.length, 33)
    eta = np.linspace(0.0, 1.0, 17)
    values = np.ones((t.size, z.size, eta.size))
    # Integral of 0.5*(Rout^2-Rin^2) dz dt, without 2*pi.
    inner = geometry.inner_radius_inlet + z/geometry.length*(
        geometry.inner_radius_outlet-geometry.inner_radius_inlet)
    expected = 2.0*np.trapezoid(0.5*((inner+0.08)**2-inner**2), z)
    np.testing.assert_allclose(
        frustum_volume_integral(values, t, z, eta, geometry), expected, rtol=1e-10
    )


def test_frustum_projector_resolves_physical_axial_and_radial_identities():
    geometry = FrustumWall(1.0, 0.25, 0.65, 0.06, 0.10)
    n = 33
    t, z, eta = np.linspace(0, 1.2, n), np.linspace(0, 1.0, n), np.linspace(0, 1, n)
    tt, zz, ee = np.meshgrid(t, z, eta, indexing="ij")
    rr = geometry.radius(z, eta)[None, :, :]
    zeros = np.zeros_like(tt)
    for u, forcing, column, truth in (
        (zz**2, -0.24*np.ones_like(tt), 0, 0.12),
        (np.broadcast_to(rr**2, tt.shape), -0.28*np.ones_like(tt), 1, 0.07),
    ):
        matrix, target = project_frustum_pde_weak(
            t=t, z=z, eta=eta, geometry=geometry, u=u, forcing=forcing,
                reaction_features=zeros[..., None], window_shape=(21, 21, 33),
                stride_shape=(4, 4, 4),
        )
        estimate = float(matrix[:, column]@target/(matrix[:, column]@matrix[:, column]))
        np.testing.assert_allclose(estimate, truth, rtol=0.03, atol=0.002)
