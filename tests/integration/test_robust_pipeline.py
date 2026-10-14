import numpy as np
import pytest

from wphid.types import Candidate, Trajectory


def _trajectory(condition_id: str, rate: float) -> Trajectory:
    time = np.linspace(0.0, 12.0, 161)
    damage = np.exp(-rate * time)
    return Trajectory(
        t=time,
        states={
            "Do": damage,
            "T": 700.0 + 40.0 * np.sin(time),
        },
        condition_id=condition_id,
    )


def _profiles() -> dict[float, tuple[Candidate, ...]]:
    def distractor(states):
        return np.asarray(states["T"])

    def library(signal):
        return (
            Candidate(
                name="damage",
                evaluate=signal,
                dimension=(0,),
                coefficient_dimension=(0,),
                group="damage",
            ),
            Candidate(
                name="temperature",
                evaluate=distractor,
                dimension=(0,),
                coefficient_dimension=(0,),
                group="temperature",
            ),
        )

    return {
        1200.0: library(lambda states: np.asarray(states["Do"]) ** 2),
        1800.0: library(lambda states: np.asarray(states["Do"])),
    }


def test_robust_profiled_pipeline_bootstraps_training_only_and_freezes_before_simex():
    """Passing validation into bootstrap/profiling or SIMEX changes frozen evidence."""
    from wphid.pipeline import discover_robust_profiled_closed

    train_ids = ("train-a", "train-b", "train-c")
    trajectories = [
        _trajectory("train-a", 0.25),
        _trajectory("train-b", 0.32),
        _trajectory("train-c", 0.40),
        _trajectory("validation-sentinel", 0.55),
    ]
    callback_calls = []

    def reconstruct(level, replicate, reconstruction_train_ids, candidates):
        callback_calls.append(
            (level, replicate, reconstruction_train_ids, tuple(item.name for item in candidates))
        )
        coordinate = np.linspace(-1.0, 1.0, 80)
        perturbation = level * 0.01 * np.sin((replicate + 1) * coordinate)
        design = np.column_stack((coordinate + perturbation, coordinate**2))
        target = 2.0 * coordinate + 0.5 * perturbation
        return design, target

    result = discover_robust_profiled_closed(
        trajectories=trajectories,
        candidate_profiles=_profiles(),
        parameter_name="theta",
        target_dimension=(0,),
        train_ids=train_ids,
        validation_ids=("validation-sentinel",),
        thresholds=(0.01,),
        n_resamples=7,
        block_length=2,
        random_seed=17,
        simex_levels=(0.0, 0.5, 1.0),
        simex_replicates=2,
        observation_noise_reconstructor=reconstruct,
        simex_extrapolation_condition_limit=1.0,
    )

    assert result.support.tolist() == [0]
    assert result.coefficients[0] == pytest.approx(2.0)
    assert callback_calls
    assert {call[2] for call in callback_calls} == {train_ids}
    assert all("validation-sentinel" not in call[2] for call in callback_calls)
    assert result.diagnostics["inclusion_probability"]["damage"] == pytest.approx(1.0)
    assert result.diagnostics["sign_consistency"]["damage"] == pytest.approx(1.0)
    assert sum(result.diagnostics["theta_frequency"].values()) == pytest.approx(1.0)
    assert result.diagnostics["selected_parameter"] == pytest.approx(1800.0)
    assert result.diagnostics["support_stable"] is True
    assert result.diagnostics["fixed_support_identifiable"] is True
    assert result.diagnostics["simex_extrapolation_stable"] is False
    assert result.diagnostics["stable"] is False
    assert result.diagnostics["simex_fallback"] is True
    assert result.diagnostics["validation_data_role"] == "frozen_model_evaluation_only"


def test_robust_api_has_no_test_condition_argument():
    """Adding test data to the discovery surface would violate the frozen-model boundary."""
    from inspect import signature

    from wphid.pipeline import discover_robust_profiled_closed

    assert "test_ids" not in signature(discover_robust_profiled_closed).parameters
