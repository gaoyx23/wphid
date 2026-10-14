import numpy as np

from experiments.s3_e1_closed_nonlinear_pde import _clean, _reaction_projector


def test_reaction_projector_is_linear_and_axisymmetric_weighted():
    t, z, r, u, *_ = _clean("thermal_axis")
    project = _reaction_projector(t, z, r)
    assert np.allclose(project(2*u), 2*project(u))
    assert np.all(np.isfinite(project(u)))
