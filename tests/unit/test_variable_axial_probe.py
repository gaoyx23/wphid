import numpy as np

from wphid.weak.axisymmetric_library import (
    project_axisymmetric_pde_weak,
    project_variable_axial_diffusion_probe,
)


def test_constant_coefficient_probe_matches_standard_axial_column():
    t = np.linspace(0.0, 1.0, 15)
    z = np.linspace(-0.5, 0.8, 15)
    r = np.linspace(0.0, 1.0, 15)
    tt, zz, rr = np.meshgrid(t, z, r, indexing="ij")
    field = 2.0 + 0.3 * tt + 0.7 * zz**2 + 0.2 * rr**2
    matrix, _ = project_axisymmetric_pde_weak(
        t=t, z=z, r=r, u=field, forcing=np.zeros_like(field),
        reaction_features=field[..., None], window_shape=(15, 15, 15),
        stride_shape=(5, 5, 5),
    )
    probe = project_variable_axial_diffusion_probe(
        t=t, z=z, r=r, u=field, coefficient=np.ones_like(z),
        coefficient_derivative=np.zeros_like(z),
        window_shape=(15, 15, 15), stride_shape=(5, 5, 5),
    )
    assert np.allclose(probe, matrix[:, 0], rtol=1e-12, atol=1e-12)
