import numpy as np
import pytest
from wphid.weak.polynomial_tests import PolynomialTestProjector, modulated_test_weights
from wphid.weak.moment_corrected import MomentCorrectedAxisymmetricProjector


def test_original_operators_retained_exactly_and_no_new_nodes():
    options = dict(t=np.linspace(0, 1, 9), z=np.linspace(0, 1, 9), r=np.linspace(.1, .3, 9),
                   window_shape=(9, 9, 9), stride_shape=(4, 4, 4), degree=4)
    old = MomentCorrectedAxisymmetricProjector(**options)
    new = PolynomialTestProjector(**options)
    for name in old.operators:
        assert new.operators[name].shape == (4, 729)
        np.testing.assert_array_equal(new.operators[name][:1].toarray(), old.operators[name].toarray())


@pytest.mark.parametrize("kind", ["first", "second", "radial_diffusion"])
def test_modulated_derivative_weights_annihilate_constant_field(kind):
    nodes = np.linspace(.1, .5, 9)
    weights = modulated_test_weights(nodes, kind, 1)
    assert abs(weights.sum()) < 1e-11


def test_weak_derivative_integration_by_parts_for_polynomial_fields():
    nodes = np.linspace(.1, .5, 9)
    mass = modulated_test_weights(nodes, "mass", 1)
    first = modulated_test_weights(nodes, "first", 1)
    second = modulated_test_weights(nodes, "second", 1)
    np.testing.assert_allclose(first@nodes**3, -mass@(3*nodes**2), atol=1e-12)
    np.testing.assert_allclose(second@nodes**4, mass@(12*nodes**2), atol=1e-11)
