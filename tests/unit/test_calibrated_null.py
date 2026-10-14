import numpy as np
import pytest
from wphid.weak.empirical_noise import noise_metric
from wphid.weak.calibrated_null import candidate_null_set


def model(name, baseline, configuration="same"):
    return dict(candidate_id=name, candidate_simulated_clean_baseline=[baseline],
        noise_metric=noise_metric(center=[.5], covariance=[[1.]]),
        calibrated_squared_radius=4., observation_configuration_id=configuration,
        null_simulation_source=name)


def test_candidate_uses_own_baseline_not_observation_oracle():
    result = candidate_null_set(residuals={"a":[10.5], "b":[10.5]},
        null_models={"a":model("a", 10.), "b":model("b", 0.)})
    assert result["candidate_set"] == ["a"]
    assert result["scores"]["b"]["score"] == pytest.approx(100.)


def test_all_candidates_required_no_silent_deletion():
    with pytest.raises(ValueError):
        candidate_null_set(residuals={"a":[0.]}, null_models={"a":model("a", 0.), "b":model("b", 0.)})


def test_observation_configuration_mismatch_rejected():
    with pytest.raises(ValueError, match="configurations"):
        candidate_null_set(residuals={"a":[0.], "b":[0.]},
            null_models={"a":model("a", 0.), "b":model("b", 0., "different")})


def test_no_candidate_is_a_valid_model_inadequacy_output():
    result = candidate_null_set(residuals={"a":[100.]}, null_models={"a":model("a", 0.)})
    assert result["candidate_set"] == []
    assert result["interpretation"] == "empty_model_inadequate"
