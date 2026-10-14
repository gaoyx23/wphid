"""Artificial counterexamples; none are scientific precision samples."""

import numpy as np
import pytest

from wphid.source_geometry_v6 import (
    independent_calibration_information, nested_nuisance_spaces,
    profile_information, weak_source_columns,
)


def test_known_coupling_removes_an_actual_compensation_direction():
    derivative = np.array([[1.], [-1.]])
    free = np.eye(2)
    coupled = np.ones((2, 1))
    assert nested_nuisance_spaces(free, coupled)
    assert profile_information(derivative, free).rank == 0
    known = profile_information(derivative, coupled)
    assert known.rank == 1
    np.testing.assert_allclose(known.information, [[2.]])


def test_unknown_conversion_can_restore_the_removed_compensation():
    derivative = np.array([[1.], [-1.]])
    nuisance = np.ones((2, 1))
    unknown = independent_calibration_information(
        derivative, nuisance, derivative, np.empty((0, 1)))
    assert unknown.rank == 0
    np.testing.assert_allclose(unknown.information, 0., atol=1e-25)


def test_finite_independent_calibration_matches_the_schur_complement():
    derivative = np.array([[1.], [-1.]])
    nuisance = np.ones((2, 1))
    # Information 2 - 2^2/(2 + 3^2), with finite assay precision 3.
    calibrated = independent_calibration_information(
        derivative, nuisance, derivative, np.array([[3.]]))
    np.testing.assert_allclose(calibrated.information, [[18./11.]], rtol=1e-12)
    assert calibrated.rank == 1
    assert calibrated.information[0, 0] < 2.


def test_wrong_fixed_loading_can_produce_zero_residual_and_kinetic_bias():
    derivative = np.array([[1.], [-1.]])
    nuisance = np.ones((2, 1))
    diagnostic = profile_information(derivative, nuisance)
    calibration_error = .2 * derivative
    projected_error = profile_information(calibration_error, nuisance).residual_derivative
    kinetic_shift = -np.linalg.pinv(diagnostic.residual_derivative) @ projected_error
    np.testing.assert_allclose(kinetic_shift, [[-.2]], atol=1e-12)
    np.testing.assert_allclose(derivative @ kinetic_shift + calibration_error, 0., atol=1e-12)


def test_fewer_nuisances_need_not_provide_a_strict_information_gain():
    derivative = np.array([[0.], [0.], [1.]])
    free = np.eye(3)[:, :2]
    coupled = np.array([[1.], [1.], [0.]])
    assert nested_nuisance_spaces(free, coupled)
    np.testing.assert_allclose(profile_information(derivative, free).information,
                               profile_information(derivative, coupled).information)


def test_natural_conditions_separate_only_with_the_declared_global_nuisance():
    assert profile_information([[1.]], [[1.]]).rank == 0
    assert profile_information([[-1.]], [[1.]]).rank == 0
    derivative = np.array([[1.], [-1.]])
    global_nuisance = np.ones((2, 1))
    assert profile_information(derivative, global_nuisance).rank == 1
    assert profile_information(derivative, np.eye(2)).rank == 0


def test_information_is_invariant_under_row_units_and_their_metric():
    derivative = np.array([[1., 0.], [0., 2.], [1., 1.]])
    nuisance = np.array([[1.], [2.], [-1.]])
    whitener = np.array([[1., .2, 0.], [0., 2., -.1], [0., 0., .5]])
    units = np.diag([1000., .01, 4.])
    original = profile_information(derivative, nuisance, whitener=whitener)
    transformed = profile_information(units @ derivative, units @ nuisance,
                                      whitener=whitener @ np.linalg.inv(units))
    np.testing.assert_allclose(original.information, transformed.information, rtol=1e-12)


def test_nuisance_units_do_not_change_the_projection():
    derivative = np.array([[1.], [2.], [3.]])
    nuisance = np.array([[1., 0.], [1., 1.], [0., 1.]])
    reference = profile_information(derivative, nuisance)
    changed = profile_information(derivative, nuisance * np.array([1e-12, 1e12]))
    np.testing.assert_allclose(reference.information, changed.information, rtol=1e-12)
    assert reference.nuisance_rank == changed.nuisance_rank == 2


def test_parameter_units_and_unobserved_parameter_directions_are_explicit():
    derivative = np.array([[1., 2.]])
    diagnostic = profile_information(derivative, np.empty((1, 0)), parameter_scale=[2., 3.])
    np.testing.assert_allclose(diagnostic.information, [[4., 12.], [12., 36.]])
    assert diagnostic.rank == 1
    assert diagnostic.smallest_singular_value == 0.
    np.testing.assert_allclose(np.array([[2., 6.]]) @ diagnostic.parameter_null_basis,
                               0., atol=1e-12)


def test_zero_activity_never_gains_rank_from_numerical_regularization():
    diagnostic = profile_information(np.zeros((3, 2)), np.zeros((3, 1)))
    assert diagnostic.rank == 0
    assert diagnostic.nuisance_rank == 0
    np.testing.assert_array_equal(diagnostic.information, np.zeros((2, 2)))
    assert diagnostic.parameter_null_basis.shape == (2, 2)


def test_state_loading_must_be_multiplied_before_weak_integration():
    weights = np.array([[[.5, .5]], [[.25, .75]]])
    rates = np.array([[1., 3.]])
    loadings = np.array([[[1., 2.]], [[-1., -1.]]])
    actual = weak_source_columns(weights, rates, loadings)
    np.testing.assert_allclose(actual, [[3.5], [-2.5]])
    # Averaging state-dependent conversion first gives 1.5*2=3, a different model.
    assert actual[0, 0] != np.mean(loadings[0, 0]) * np.mean(rates[0])


def test_non_nested_constraint_does_not_qualify_for_the_gain_inequality():
    free = np.array([[1.], [0.]])
    restricted = np.array([[0.], [1.]])
    assert not nested_nuisance_spaces(free, restricted)
    derivative = restricted
    assert profile_information(derivative, free).rank == 1
    assert profile_information(derivative, restricted).rank == 0


@pytest.mark.parametrize("metric", [np.diag([1., 0.]), np.array([[1., np.nan], [0., 1.]])])
def test_invalid_metric_cannot_hide_an_observation_direction(metric):
    with pytest.raises(ValueError):
        profile_information(np.ones((2, 1)), np.empty((2, 0)), whitener=metric)
