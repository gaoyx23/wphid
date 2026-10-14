"""Independent repair of a floating exact-equality assertion; kernel unchanged."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from wphid.coupled_balance_v1.noise import SharedObservationLinearization, observation_factor


def test_sparse_covariance_action_respects_roundoff_bound_and_dense_cap():
    linearization = SharedObservationLinearization(csr_matrix(np.eye(3)), (), ("a", "b", "c"))
    noise = linearization.noise_operator([], observation_factor([1., 2., 3.]))
    expected = np.array([1., 2., 3.])
    # A square root and two products introduce roundoff; bitwise equality is
    # invalid for sqrt(2)^2. This bound concerns numerical arithmetic only.
    epsilon = np.finfo(float).eps
    bound = 8*epsilon/(1-8*epsilon)
    np.testing.assert_allclose(noise.matvec([1., 1., 1.]), expected, rtol=bound, atol=0.)
    with pytest.raises(ValueError, match="cap"):
        noise.dense(max_rows=2)


def test_joint_coordinate_units_preserve_covariance_actions():
    jacobian = np.array([[1., 1., 0.], [0., 1., 1.]])
    variances = np.array([1., 4., 9.])
    units = np.array([1e-6, 2e3, 4.])
    original = SharedObservationLinearization(jacobian, (), ("a", "b", "c"))
    changed = SharedObservationLinearization(jacobian/units, (), ("a", "b", "c"))
    first = original.noise_operator([], observation_factor(variances))
    second = changed.noise_operator([], observation_factor(variances*units**2))
    vector = np.array([.7, -1.3])
    np.testing.assert_allclose(first.matvec(vector), second.matvec(vector), rtol=1e-14, atol=1e-14)
