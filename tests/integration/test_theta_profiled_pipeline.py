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


def _library_builder(theta: float):
    """The candidate matches the target only for theta >= 1500 (signal = Do),
    so the profile likelihood must escape the lower grid point 1200."""
    def distractor(states):
        return np.asarray(states["T"])

    def signal(states):
        damage = np.asarray(states["Do"])
        return damage**2 if theta < 1500.0 else damage

    return (
        Candidate(
            name=f"damage(theta={theta})",
            evaluate=signal,
            dimension=(0,),
            coefficient_dimension=(0,),
            group="damage",
            canonical_identity="damage",
        ),
        Candidate(
            name="temperature",
            evaluate=distractor,
            dimension=(0,),
            coefficient_dimension=(0,),
            group="temperature",
            canonical_identity="temperature",
        ),
    )


def test_theta_pipeline_selects_support_then_freezes_validation_before_simex():
    """Support and theta come from training-only weak data; validation enters
    only as frozen-model evaluation and never reaches the SIMEX reconstructor."""
    from wphid.pipeline import discover_theta_profiled_closed

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

    result = discover_theta_profiled_closed(
        trajectories=trajectories,
        theta_grid=(1200.0, 1800.0),
        library_builder=_library_builder,
        parameter_name="theta",
        target_dimension=(0,),
        train_ids=train_ids,
        validation_ids=("validation-sentinel",),
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
    assert result.diagnostics["selected_parameter"] > 1450.0
    assert result.diagnostics["theta_selection_data"] == "training_conditions_only"
    assert result.diagnostics["theta_selection_rule"] == (
        "best_subset_aic_grid_profile_likelihood_alternating"
    )
    assert result.diagnostics["validation_data_role"] == "frozen_model_evaluation_only"
    assert result.diagnostics["support_stable"] is True
    assert result.diagnostics["normalize_per_condition"] is True
    assert "aic_history" in result.diagnostics
    assert "refinement_history" in result.diagnostics
    assert callback_calls
    assert {call[2] for call in callback_calls} == {train_ids}
    assert all("validation-sentinel" not in call[2] for call in callback_calls)


def test_theta_pipeline_api_has_no_test_condition_argument():
    """Adding test data to the discovery surface would violate the frozen-model boundary."""
    from inspect import signature

    from wphid.pipeline import discover_theta_profiled_closed

    assert "test_ids" not in signature(discover_theta_profiled_closed).parameters
