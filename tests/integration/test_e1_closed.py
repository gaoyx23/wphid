import json

import numpy as np
import pytest

from wphid.pipeline import discover_closed
from wphid.truth.aging_ode import simulate_named_condition
from wphid.types import Trajectory


EXPECTED_SUPPORT = {
    "arr(T;theta=1800)*pO2",
    "arr(T;theta=1800)*pO2*sigma",
    "arr(T;theta=1800)*pO2*Df",
}


def _closed_candidate_library():
    from experiments.e1_closed_library import closed_candidate_library

    return closed_candidate_library()


def test_closed_pipeline_recovers_three_oxidation_terms_without_spurious_support():
    """Skipping validation selection, weak fitting, or sparsity breaks E1 recovery."""
    condition_ids = ("thermal_hold", "fatigue_rich", "oxygen_rich")
    trajectories = [simulate_named_condition(name) for name in condition_ids]

    result = discover_closed(
        trajectories=trajectories,
        candidates=_closed_candidate_library(),
        train_ids=("thermal_hold", "fatigue_rich"),
        validation_ids=("oxygen_rich",),
        thresholds=np.geomspace(1e-4, 0.5, 48),
    )

    selected_names = {result.names[index] for index in result.support}
    assert selected_names == EXPECTED_SUPPORT
    assert result.validation_error < 0.03
    assert result.diagnostics["normalized_rmse_definition"] == "RMS(residual)/RMS(target)"


def test_closed_pipeline_rejects_duplicate_training_condition_ids():
    """Repeating a training ID would silently overweight one condition."""
    trajectories = [
        simulate_named_condition(name)
        for name in ("thermal_hold", "fatigue_rich", "oxygen_rich")
    ]

    with pytest.raises(ValueError, match="train_ids must contain unique"):
        discover_closed(
            trajectories=trajectories,
            candidates=_closed_candidate_library(),
            train_ids=("thermal_hold", "thermal_hold"),
            validation_ids=("oxygen_rich",),
            thresholds=(0.01,),
        )


def test_closed_pipeline_rejects_duplicate_validation_condition_ids():
    """Repeating a validation ID would silently overweight held-out error."""
    trajectories = [
        simulate_named_condition(name)
        for name in ("thermal_hold", "fatigue_rich", "oxygen_rich")
    ]

    with pytest.raises(ValueError, match="validation_ids must contain unique"):
        discover_closed(
            trajectories=trajectories,
            candidates=_closed_candidate_library(),
            train_ids=("thermal_hold", "fatigue_rich"),
            validation_ids=("oxygen_rich", "oxygen_rich"),
            thresholds=(0.01,),
        )


def test_e1_experiment_writes_fixed_configuration_metrics(tmp_path):
    """Dropping reproducible configuration or metrics persistence breaks E1."""
    from experiments.e1_closed_library import run_experiment

    metrics = run_experiment(tmp_path)
    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert persisted == metrics
    assert metrics["seed"] == 20260729
    assert metrics["configuration"]["train_ids"] == [
        "thermal_hold",
        "fatigue_rich",
    ]
    assert metrics["configuration"]["validation_ids"] == ["oxygen_rich"]
    assert set(metrics["support"]) == EXPECTED_SUPPORT
    assert metrics["validation_normalized_rmse"] < 0.03
    assert metrics["configuration"]["sparsity_penalty_per_term"] == 0.001
    assert metrics["configuration"]["normalized_rmse_definition"] == (
        "RMS(residual)/RMS(target)"
    )


def test_e1_library_uses_shared_parameterized_arrhenius_identity():
    """A local generic arr(T) evaluator would conflate distinct physics."""
    from experiments.e1_closed_library import closed_candidate_library

    names = {candidate.name for candidate in closed_candidate_library()}

    assert {
        "arr(T;theta=1800)*pO2",
        "arr(T;theta=1800)*pO2*sigma",
        "arr(T;theta=1800)*pO2*Df",
    } <= names
    assert all(name != "arr(T)" and not name.startswith("arr(T)*") for name in names)


def test_closed_pipeline_reports_effective_short_trajectory_weak_settings():
    """Reporting defaults would hide reduced weak windows and strides."""
    trajectories = []
    for condition_id in ("thermal_hold", "fatigue_rich", "oxygen_rich"):
        full = simulate_named_condition(condition_id)
        trajectories.append(
            Trajectory(
                t=full.t[:11],
                states={name: values[:11] for name, values in full.states.items()},
                condition_id=condition_id,
            )
        )

    result = discover_closed(
        trajectories=trajectories,
        candidates=_closed_candidate_library(),
        train_ids=("thermal_hold", "fatigue_rich"),
        validation_ids=("oxygen_rich",),
        thresholds=(0.01,),
    )

    expected_by_condition = {
        "thermal_hold": 11,
        "fatigue_rich": 11,
        "oxygen_rich": 11,
    }
    expected_stride_by_condition = {
        "thermal_hold": 10,
        "fatigue_rich": 10,
        "oxygen_rich": 10,
    }
    assert result.diagnostics["weak_window"] == 11
    assert result.diagnostics["weak_stride"] == 10
    assert result.diagnostics["weak_window_by_condition"] == expected_by_condition
    assert result.diagnostics["weak_stride_by_condition"] == expected_stride_by_condition
