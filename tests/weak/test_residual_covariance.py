import numpy as np
import pytest
from scipy.sparse import csr_matrix
from wphid.weak.residual_covariance import linearized_residual_covariance, covariance_cholesky


def test_residual_covariance_matches_finite_difference_of_noisy_balance():
    rng = np.random.default_rng(9)
    operators = {k: csr_matrix(rng.normal(size=(3, 6))) for k in ("mass", "time", "axial", "radial")}
    states = np.array([np.linspace(1, 2, 6), np.linspace(0.4, 0.7, 6)])
    base, amplitude = np.array([0.1, 0.2, -0.3]), -2.0
    stds = [0.02, 0.01]
    gradients = [states[1], states[0]]
    def residual(s):
        u = s[1]
        return (-operators["time"]@u-base[0]*(operators["axial"]@u)
                -base[1]*(operators["radial"]@u)-base[2]*(operators["mass"]@u)
                -amplitude*(operators["mass"]@(s[0]*s[1])))
    jacobian = []
    for field in range(2):
        for node in range(6):
            a, b = states.copy(), states.copy()
            a[field, node] += 1e-6
            b[field, node] -= 1e-6
            jacobian.append((residual(a)-residual(b))/(2e-6)*stds[field])
    jacobian = np.column_stack(jacobian)
    predicted = linearized_residual_covariance(operators, base, amplitude, gradients, stds, 1)
    np.testing.assert_allclose(predicted, jacobian@jacobian.T, rtol=1e-8, atol=1e-12)
    assert abs(predicted[0, 1]) > 1e-5


def test_fixed_cholesky_is_reconstructible_and_zero_noise_is_rejected():
    matrix = np.array([[1.0, 0.3], [0.3, 2.0]])
    factor, info = covariance_cholesky(matrix)
    np.testing.assert_allclose(factor@factor.T, matrix+info["absolute_jitter"]*np.eye(2))
    with pytest.raises(ValueError):
        covariance_cholesky(np.zeros((2, 2)))
