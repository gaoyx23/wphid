import numpy as np
import pytest
from experiments.audit_coupled_reaction_compensation import CASES, discrepancy


def test_pilot_has_no_both_present_duplicate_and_nonnegative_rates():
    assert len(CASES) == 7 and len({c[0] for c in CASES}) == 7
    assert all(o >= 0 and s >= 0 and o*s == 0 for _,o,s in CASES)


def test_identical_prediction_zero_and_metric_is_explicitly_oracle():
    result = discrepancy(np.arange(1., 13.), np.arange(1., 13.))
    assert result["oracle_noise_weighted_squared_bias"] == 0
    assert result["calibrated_candidate_decision"] is None
    assert result["oracle_metric_not_available_to_engineering_inference"]


def test_woodbury_matches_dense_covariance():
    actual = np.arange(1., 13.)
    predicted = actual + np.linspace(-.1, .1, 12)
    residual = (predicted-actual)/(.01*actual)
    modes = np.column_stack([np.ones(12), np.r_[np.ones(4),np.zeros(8)]])
    expected = residual @ np.linalg.solve(np.eye(12)+modes@modes.T, residual)
    assert discrepancy(actual,predicted)["oracle_noise_weighted_squared_bias"] == pytest.approx(expected)


def test_invalid_data_not_clipped():
    with pytest.raises(ValueError): discrepancy(np.array([0.,1.,1.]),np.ones(3))
