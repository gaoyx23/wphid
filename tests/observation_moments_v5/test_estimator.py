"""Artificial counterexamples and exact small-model verification."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from wphid.observation_moments_v5.estimator import CommonCriterion, Model, RankFailure, fit_model, select_models, tangent_diagnostics
from wphid.observation_moments_v5.jets import VectorJet, corrected_inner, family_derivatives
from wphid.final_candidate_v1.problem import paper_families


def test_product_cross_derivative_is_required_for_linear_noise():
    identity = csr_matrix(np.eye(2))
    left = VectorJet(np.array([2., 3.]), (identity,), (np.zeros(2),))
    right = VectorJet(np.array([4., 6.]), (2*identity,), (np.zeros(2),))
    assert corrected_inner(left, right, [.25]) == 25.
    assert left.value @ right.value == 26.


def test_product_curvature_and_cross_terms_are_both_present():
    # Scalar h(y)=y^2, u(y)=y: (h*u)''=6y, so correction is y^3-3*sigma^2*y.
    left = VectorJet(np.array([4.]), (csr_matrix([[4.]]),), (np.array([2.]),))
    right = VectorJet(np.array([2.]), (csr_matrix([[1.]]),), (np.array([0.]),))
    assert corrected_inner(left, right, [.5]) == 5.


def test_correlated_weight_profile_matches_direct_quadratic():
    covariance = np.array([[2., .4, .1], [.4, 1., -.1], [.1, -.1, .8]])
    target = np.array([2., 1., 4.])
    matrix = np.array([[1., 0.], [1., 1.], [1., 3.]])
    criterion = CommonCriterion(target, ('a', 'b', 'c'), covariance)
    actual = criterion.profile(matrix)
    inverse = np.linalg.inv(covariance)
    expected = np.linalg.solve(matrix.T @ inverse @ matrix, matrix.T @ inverse @ target)
    np.testing.assert_allclose(actual['coefficients'], expected, rtol=1e-12)
    residual = target-matrix @ expected
    np.testing.assert_allclose(actual['objective'], residual @ inverse @ residual, rtol=1e-12)


def test_rank_failure_is_retained_and_never_ridged():
    criterion = CommonCriterion([1., 2., 3.], ('a', 'b', 'c'))
    model = Model('bad', (), 2, lambda p: np.ones((3, 2)))
    result = fit_model(criterion, model)
    assert result['status'] == 'failed_retained'
    assert result['profile'] is None


def test_just_identified_moments_cannot_identify_kinetics():
    criterion = CommonCriterion([2., 3.], ('a', 'b'))
    model = Model('degenerate', ((.1, 3.),), 2, lambda p: np.array([[1., p[0]], [0., 1.]]))
    assert criterion.profile(model.matrix([.4]))['objective'] < 1e-25
    assert criterion.profile(model.matrix([2.]))['objective'] < 1e-25
    diagnostics = tangent_diagnostics(criterion, model, [.4])
    np.testing.assert_allclose(diagnostics['tangent'], 0., atol=1e-12)
    assert diagnostics['rank'] == 0


def test_overidentified_profile_recovers_nonlinear_parameter():
    coordinate = np.array([0., .3, .7, 1.1, 1.6, 2.])
    matrix = lambda p: np.column_stack((np.ones(len(coordinate)), np.exp(-p[0]*coordinate)))
    target = matrix([.8]) @ np.array([1.2, 2.3])
    criterion = CommonCriterion(target, tuple(map(str, range(len(coordinate)))))
    model = Model('overidentified', ((.2, 1.8),), 2, matrix)
    fit = fit_model(criterion, model, max_evaluations=300)
    assert fit['status'] == 'completed'
    np.testing.assert_allclose(fit['parameters'], [.8], atol=2e-6)
    np.testing.assert_allclose(fit['profile']['coefficients'], [1.2, 2.3], atol=1e-5)
    assert tangent_diagnostics(criterion, model, fit['parameters'])['rank'] == 1


def test_common_penalty_selects_minimal_nested_model_and_includes_null():
    target = np.array([1., 2., 3., 4.])
    criterion = CommonCriterion(target, ('a', 'b', 'c', 'd'))
    models = (Model('null', (), 0, lambda p: np.empty((4, 0))),
              Model('small', (), 1, lambda p: target[:, None]),
              Model('larger', (), 2, lambda p: np.column_stack((target, np.ones(4)))))
    result = select_models(criterion, models, .1)
    assert result['selected_key'] == 'small'
    assert result['all_declared_candidates_completed']
    assert len(result['records']) == 3
    assert all(r['profile']['objective'] >= 0 for r in result['records'])


def test_failed_candidate_is_not_removed_from_search_qualification():
    criterion = CommonCriterion([1., 2., 3.], ('a', 'b', 'c'))
    models = (Model('valid', (), 1, lambda p: np.arange(1., 4.)[:, None]),
              Model('invalid', (), 2, lambda p: np.ones((3, 2))))
    assert not select_models(criterion, models, .1)['all_declared_candidates_completed']


def test_evaluation_cap_applies_to_all_restarts_together():
    coordinate = np.linspace(0., 1., 8)
    matrix = lambda p: np.exp(-p[0]*coordinate)[:, None]
    criterion = CommonCriterion(matrix([.8])[:, 0], tuple(map(str, range(8))))
    result = fit_model(criterion, Model('limited', ((.2, 2.),), 1, matrix), max_evaluations=2)
    assert result['status'] == 'budget_exhausted'
    assert result['objective_evaluations'] == 2
    assert result['profile'] is not None


@pytest.mark.parametrize('family', paper_families(), ids=lambda f: f.name)
def test_all_family_state_derivatives_against_finite_differences(family):
    states = [np.array([1000., 1300.]), np.array([.7, 1.3]), np.array([.2, .6])]
    _, first, second = family_derivatives(family, states, family.center)
    for axis, step in enumerate((.1, 1e-4, 1e-4)):
        plus, minus = [s.copy() for s in states], [s.copy() for s in states]
        plus[axis] += step
        minus[axis] -= step
        center = family.values(states, family.center)
        high, low = family.values(plus, family.center), family.values(minus, family.center)
        np.testing.assert_allclose(first[axis], (high-low)/(2*step), rtol=2e-6, atol=2e-10)
        np.testing.assert_allclose(second[axis], (high-2*center+low)/(step*step), rtol=2e-5, atol=2e-8)


def test_nonpositive_weight_is_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        CommonCriterion([1., 2.], ('a', 'b'), np.array([[1., 2.], [2., 1.]]))
