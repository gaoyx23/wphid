import numpy as np
from reproducibility.diagnose_h3_likelihood_stationarity import projected_gradient


def test_box_kkt_only_removes_outward_descent_components_at_bounds():
    vector = np.array([0., 1., 0., 1., .5, 0.])
    gradient = np.array([2., -2., -2., 2., 3., 4.])
    np.testing.assert_array_equal(projected_gradient(vector, gradient, 5), [0., 0., -2., 2., 3., 4.])
    np.testing.assert_array_equal(gradient, [2., -2., -2., 2., 3., 4.])
