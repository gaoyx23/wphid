import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from wphid.weak.moment_corrected import corrected_test_weights, MomentCorrectedAxisymmetricProjector


@pytest.mark.parametrize("kind", ["mass", "first", "second", "radial_mass", "radial_diffusion"])
def test_all_low_order_functional_moments(kind):
    x = np.linspace(0.3, 1.2, 15)
    weights, _ = corrected_test_weights(x, kind)
    center, half = (x[0]+x[-1])/2, (x[-1]-x[0])/2
    q, gauss = leggauss(16)
    nodes = center + half*q
    b = 1-q*q
    value, first = b**4, -8*q*b**3/half
    second = -8*b*b*(1-7*q*q)/half**2
    test = {"mass": value, "first": first, "second": second,
            "radial_mass": nodes*value, "radial_diffusion": nodes*second+first}[kind]
    for degree in range(5):
        assert weights @ x**degree == pytest.approx(half*gauss @ (test*nodes**degree), abs=1e-12)


def test_polynomial_pde_balance_is_exact_without_truth_correction():
    t, z, r = np.linspace(0, 1, 11), np.linspace(-0.5, 1, 11), np.linspace(0, 1, 11)
    tt, zz, rr = np.meshgrid(t, z, r, indexing="ij")
    u = 2 + tt*zz**2 + tt**2*rr**2 + 0.2*zz*rr**2
    ut = zz**2 + 2*tt*rr**2
    uzz = 2*tt
    radial = 4*tt**2 + 0.8*zz
    truth = np.array([0.13, 0.07, -0.3])
    forcing = ut - truth[0]*uzz - truth[1]*radial - truth[2]*u
    projector = MomentCorrectedAxisymmetricProjector(t=t, z=z, r=r,
                         window_shape=(7, 7, 7), stride_shape=(4, 4, 4))
    matrix, target = projector.library(u, forcing)
    np.testing.assert_allclose(matrix @ truth, target, atol=2e-15)
    assert matrix.shape == (8, 3)
    with pytest.raises(ValueError):
        projector.mass(u[:-1])


def test_constant_has_zero_derivative_even_at_axis():
    x = np.linspace(0, 1, 15)
    for kind in ("first", "second", "radial_diffusion"):
        weights, _ = corrected_test_weights(x, kind)
        assert abs(weights.sum()) < 2e-14
    with pytest.raises(ValueError):
        corrected_test_weights(x, "unsupported")
