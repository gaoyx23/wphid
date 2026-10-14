"""Objective/logdet/gradient qualifications, not parameter fitting."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from wphid.coupled_balance_v1.assembler import WeakSystem
from wphid.coupled_balance_v1.noise import CovarianceRankFailure, SharedObservationLinearization, observation_factor
from wphid.coupled_likelihood_v1 import LikelihoodBudgetFailure, evaluate_quasi_likelihood


def fixture():
    target = np.array([.2, .7, -.3])
    design = np.array([[.3, .1], [.4, -.2], [.1, .5]])
    balance = WeakSystem(target, design, {}, (("f", 0, 0), ("f", 0, 1), ("f", 0, 2)), ("a", "b"))
    base = np.array([[1., .1, 0., .2], [.2, 1., .1, 0.], [.1, 0., 1., .3]])
    first = np.array([[.1, 0., .2, 0.], [0., .1, .1, 0.], [.2, .1, 0., 0.]])
    second = np.array([[0., .1, 0., .2], [.1, 0., .1, 0.], [0., .2, .1, .1]])
    derivative = SharedObservationLinearization(base, (first, second), ("x0", "x1", "x2", "x3"))
    factor = observation_factor([.5, 1., .8, .3], np.array([[.1], [.2], [.1], [.3]]))
    return balance, derivative, factor


def test_sparse_objective_retains_parameter_dependent_logdet_and_cross_rows():
    balance, derivative, factor = fixture()
    coefficients = np.array([.2, .3])
    value = evaluate_quasi_likelihood(balance, derivative, coefficients, factor)
    covariance = derivative.noise_operator(coefficients, factor).dense()
    residual = balance.residual(coefficients)
    expected_quad = residual @ np.linalg.solve(covariance, residual)
    expected_logdet = np.linalg.slogdet(covariance)[1]
    np.testing.assert_allclose([value.quadratic, value.log_determinant, value.objective],
                               [expected_quad, expected_logdet, .5*(expected_quad+expected_logdet)], atol=1e-14)
    assert abs(expected_logdet) > .1
    assert np.count_nonzero(covariance-np.diag(np.diag(covariance))) > 0


def test_joint_coefficient_gradient_includes_covariance_derivative_and_logdet():
    balance, derivative, factor = fixture()
    coefficients = np.array([.2, .3])
    value = evaluate_quasi_likelihood(balance, derivative, coefficients, factor, compute_linear_gradient=True)
    numerical = []
    for index in range(2):
        plus, minus = coefficients.copy(), coefficients.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical.append((evaluate_quasi_likelihood(balance, derivative, plus, factor).objective-
                          evaluate_quasi_likelihood(balance, derivative, minus, factor).objective)/2e-5)
    np.testing.assert_allclose(value.linear_gradient, numerical, rtol=2e-8, atol=2e-10)
    fixed_weight_gradient = -balance.matrix.T @ np.linalg.solve(derivative.noise_operator(coefficients, factor).dense(), balance.residual(coefficients))
    assert np.linalg.norm(value.linear_gradient-fixed_weight_gradient) > .01


def test_row_unit_change_preserves_quadratic_and_changes_only_constant_logdet():
    balance, derivative, factor = fixture()
    coefficients = [.2, .3]
    scale = np.array([1e-5, 3., 1e7])
    changed = WeakSystem(scale*balance.target, scale[:, None]*balance.matrix, {}, balance.row_keys, balance.column_keys)
    changed_derivative = SharedObservationLinearization(derivative.target_jacobian.multiply(scale[:, None]),
        tuple(column.multiply(scale[:, None]) for column in derivative.column_jacobians), derivative.coordinate_keys)
    original = evaluate_quasi_likelihood(balance, derivative, coefficients, factor, compute_linear_gradient=True)
    value = evaluate_quasi_likelihood(changed, changed_derivative, coefficients, factor, compute_linear_gradient=True)
    np.testing.assert_allclose(value.quadratic, original.quadratic, rtol=1e-12)
    np.testing.assert_allclose(value.log_determinant-original.log_determinant, 2*np.sum(np.log(scale)), atol=1e-12)
    np.testing.assert_allclose(value.linear_gradient, original.linear_gradient, rtol=1e-12, atol=1e-12)


def test_singular_shared_observation_covariance_is_rejected_without_ridge():
    balance = WeakSystem(np.ones(2), np.ones((2, 1)), {}, (("f", 0, 0), ("f", 0, 1)), ("a",))
    derivative = SharedObservationLinearization([[1.], [1.]], ([[.1], [.1]],), ("x",))
    with pytest.raises(CovarianceRankFailure):
        evaluate_quasi_likelihood(balance, derivative, [.2], eye(1, format="csr"))


def test_global_sparse_score_does_not_require_a_dense_inverse():
    rows = 700
    balance = WeakSystem(np.ones(rows), np.empty((rows, 0)), {}, tuple(("f", 0, i) for i in range(rows)), ())
    derivative = SharedObservationLinearization(eye(rows, format="csr"), (), tuple(f"x{i}" for i in range(rows)))
    value = evaluate_quasi_likelihood(balance, derivative, [], eye(rows, format="csr"))
    np.testing.assert_allclose(value.objective, rows/2)
    assert value.covariance_nnz == rows
    assert value.linear_gradient is None
    with pytest.raises(LikelihoodBudgetFailure, match="gradient cap"):
        evaluate_quasi_likelihood(balance, derivative, [], eye(rows, format="csr"), compute_linear_gradient=True)


def test_common_error_factor_pair_budget_prevents_unbounded_dense_covariance():
    rows = 20
    balance = WeakSystem(np.ones(rows), np.empty((rows, 0)), {}, tuple(("f", 0, i) for i in range(rows)), ())
    derivative = SharedObservationLinearization(eye(rows, format="csr"), (), tuple(f"x{i}" for i in range(rows)))
    with pytest.raises(LikelihoodBudgetFailure, match="before matrix"):
        evaluate_quasi_likelihood(balance, derivative, [], csr_matrix(np.ones((rows, 1))), max_covariance_pairs=100)


def test_target_design_noise_cancellation_is_retained_as_rank_failure():
    balance = WeakSystem(np.ones(2), np.ones((2, 1)), {}, (("f", 0, 0), ("f", 0, 1)), ("a",))
    derivative = SharedObservationLinearization(eye(2, format="csr"), (eye(2, format="csr"),), ("x0", "x1"))
    with pytest.raises(CovarianceRankFailure, match="diagonal"):
        evaluate_quasi_likelihood(balance, derivative, [1.], eye(2, format="csr"))
