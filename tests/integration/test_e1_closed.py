from dataclasses import replace
import json

import numpy as np
import pytest

from wphid.dimensions import DAMAGE_RATE
from wphid.pipeline import discover_closed
from wphid.truth.aging_ode import simulate_named_condition
from wphid.types import Trajectory


TEST_PROFILE_THETA = 1800.0
EXPECTED_SUPPORT = {
    "arr(T;theta=1800)*pO2",
    "arr(T;theta=1800)*pO2*sigma",
    "arr(T;theta=1800)*pO2*Df",
}


def _closed_candidate_library(
    arrhenius_temperature: float = TEST_PROFILE_THETA,
):
    from experiments.e1_closed_library import closed_candidate_library

    return closed_candidate_library(arrhenius_temperature)


def test_closed_pipeline_recovers_three_oxidation_terms_without_spurious_support():
    """Skipping validation selection, weak fitting, or sparsity breaks E1 recovery."""
    condition_ids = ("thermal_hold", "fatigue_rich", "oxygen_rich")
    trajectories = [simulate_named_condition(name) for name in condition_ids]

    result = discover_closed(
        trajectories=trajectories,
        candidates=_closed_candidate_library(),
        target_dimension=DAMAGE_RATE,
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
            target_dimension=DAMAGE_RATE,
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
            target_dimension=DAMAGE_RATE,
            train_ids=("thermal_hold", "fatigue_rich"),
            validation_ids=("oxygen_rich", "oxygen_rich"),
            thresholds=(0.01,),
        )


def test_e1_experiment_writes_fixed_configuration_metrics(tmp_path):
    """Dropping reproducible configuration or metrics persistence breaks E1."""
    from experiments.e1_closed_library import (
        CONDITION_IDS,
        NOISE_SEEDS,
        PROVENANCE_ID,
        TRAIN_IDS,
        VALIDATION_IDS,
        run_experiment,
    )

    metrics = run_experiment(tmp_path)
    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert persisted == metrics
    assert metrics["provenance_id"] == PROVENANCE_ID == 20260729
    assert "seed" not in metrics
    assert metrics["configuration"]["condition_ids"] == list(CONDITION_IDS) == [
        "thermal_hot",
        "fatigue_warm",
        "fatigue_cool",
        "fatigue_low_oxygen",
        "validation_hot_fatigue",
        "validation_cool_fatigue",
    ]
    assert metrics["configuration"]["train_ids"] == list(TRAIN_IDS) == [
        "thermal_hot",
        "fatigue_warm",
        "fatigue_cool",
        "fatigue_low_oxygen",
    ]
    assert metrics["configuration"]["validation_ids"] == list(
        VALIDATION_IDS
    ) == [
        "validation_hot_fatigue",
        "validation_cool_fatigue",
    ]
    assert metrics["aggregate"]["seeds"] == list(NOISE_SEEDS) == [
        101,
        203,
        307,
        401,
        509,
    ]
    assert metrics["configuration"]["normalized_rmse_definition"] == (
        "RMS(weak residual)/RMS(weak target)"
    )


def test_e1_library_uses_shared_parameterized_arrhenius_identity():
    """A local generic arr(T) evaluator would conflate distinct physics."""
    from experiments.e1_closed_library import closed_candidate_library

    names = {
        candidate.name
        for candidate in closed_candidate_library(TEST_PROFILE_THETA)
    }

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
        target_dimension=DAMAGE_RATE,
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


def test_closed_pipeline_rejects_missing_coefficient_dimensions():
    """Undeclared coefficient units make mixed raw feature dimensions ambiguous."""
    trajectories = [
        simulate_named_condition(name)
        for name in ("thermal_hold", "fatigue_rich", "oxygen_rich")
    ]

    missing_dimension = replace(
        _closed_candidate_library()[0],
        coefficient_dimension=None,
    )

    with pytest.raises(ValueError, match="coefficient dimension"):
        discover_closed(
            trajectories=trajectories,
            candidates=[missing_dimension],
            target_dimension=DAMAGE_RATE,
            train_ids=("thermal_hold", "fatigue_rich"),
            validation_ids=("oxygen_rich",),
            thresholds=(0.01,),
        )


def test_closed_pipeline_rejects_coefficient_units_that_do_not_reach_target():
    """A coefficient times its raw feature must match the derivative target."""
    trajectories = [
        simulate_named_condition(name)
        for name in ("thermal_hold", "fatigue_rich", "oxygen_rich")
    ]
    raw_candidate = _closed_candidate_library()[0]
    incompatible = replace(
        raw_candidate,
        coefficient_dimension=(0, 0, 0, 0),
    )

    with pytest.raises(ValueError, match="does not match target"):
        discover_closed(
            trajectories=trajectories,
            candidates=[incompatible],
            target_dimension=DAMAGE_RATE,
            train_ids=("thermal_hold", "fatigue_rich"),
            validation_ids=("oxygen_rich",),
            thresholds=(0.01,),
        )


def test_arrhenius_profile_selection_uses_training_conditions_only():
    """Changing validation observations must not change selected theta evidence."""
    from experiments.e1_closed_library import (
        DAMAGE_RATE,
        NOISE_SEEDS,
        THETA_GRID,
        THRESHOLDS,
        TRAIN_IDS,
        VALIDATION_IDS,
        candidate_profiles,
        noisy_observations,
    )
    from wphid.pipeline import discover_profiled_closed

    trajectories = noisy_observations(NOISE_SEEDS[0])
    baseline = discover_profiled_closed(
        trajectories=trajectories,
        candidate_profiles=candidate_profiles(THETA_GRID),
        parameter_name="arrhenius_theta",
        target_dimension=DAMAGE_RATE,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    validation_id = VALIDATION_IDS[0]
    changed_trajectories = [
        Trajectory(
            t=trajectory.t,
            states={
                **dict(trajectory.states),
                "Do": (
                    np.linspace(0.0, 10.0, trajectory.t.size)
                    if trajectory.condition_id == validation_id
                    else trajectory.states["Do"]
                ),
            },
            condition_id=trajectory.condition_id,
        )
        for trajectory in trajectories
    ]
    changed_validation = discover_profiled_closed(
        trajectories=changed_trajectories,
        candidate_profiles=candidate_profiles(THETA_GRID),
        parameter_name="arrhenius_theta",
        target_dimension=DAMAGE_RATE,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )

    assert baseline.diagnostics["parameter_selection_data"] == (
        "training_conditions_only"
    )
    assert changed_validation.diagnostics["selected_parameter"] == (
        baseline.diagnostics["selected_parameter"]
    )
    assert changed_validation.diagnostics["parameter_profile_history"] == (
        baseline.diagnostics["parameter_profile_history"]
    )


def test_e1_runs_five_percent_multivariate_noise_and_persists_frozen_metrics(
    tmp_path,
):
    """A clean or scalar weak test cannot satisfy the frozen E1 gate."""
    from experiments.e1_closed_library import (
        CONDITION_IDS,
        NOISE_SEEDS,
        OBSERVED_FIELDS,
        TRAIN_IDS,
        VALIDATION_IDS,
        run_experiment,
    )

    metrics = run_experiment(tmp_path)
    persisted = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )

    assert persisted == metrics
    assert metrics["configuration"]["observation_noise"] == {
        "applied_to": list(OBSERVED_FIELDS),
        "distribution": (
            "independent zero-mean Gaussian per field and time sample"
        ),
        "noise_fraction": 0.05,
        "scale_per_condition_field": (
            "noise_fraction * RMS(clean field)"
        ),
    }
    assert metrics["aggregate"]["seeds"] == list(NOISE_SEEDS)
    assert len(metrics["per_seed"]) == len(NOISE_SEEDS) == 5
    assert metrics["configuration"]["condition_ids"] == list(CONDITION_IDS)
    assert metrics["configuration"]["train_ids"] == list(TRAIN_IDS)
    assert metrics["configuration"]["validation_ids"] == list(VALIDATION_IDS)
    aggregate = metrics["aggregate"]
    observed_f1 = aggregate["minimum_support_f1"]
    expected_gate_pass = observed_f1 >= 0.9
    assert aggregate["support_f1_threshold"] == 0.9
    assert aggregate["gate"] == {
        "metric": "minimum_per_seed_support_f1",
        "criterion": "greater_than_or_equal",
        "threshold": 0.9,
        "observed": observed_f1,
        "pass": expected_gate_pass,
    }
    assert aggregate["all_seeds_pass"] == expected_gate_pass

    for seed_result in metrics["per_seed"]:
        support_metrics = seed_result["support_metrics"]
        assert support_metrics.keys() >= {
            "precision",
            "recall",
            "f1",
            "true_positives",
            "false_positives",
            "false_negatives",
        }
        assert seed_result["support_gate_pass"] == (
            support_metrics["f1"] >= 0.9
        )
        assert seed_result["coefficient_metrics"].keys() >= {
            "relative_l2_error",
            "relative_absolute_error_by_term",
        }
        assert np.isfinite(
            seed_result["coefficient_metrics"]["relative_l2_error"]
        )
        assert np.isfinite(seed_result["selected_theta"])
        assert np.isfinite(seed_result["theta_relative_error"])
        assert seed_result["parameter_selection_data"] == (
            "training_conditions_only"
        )
        assert set(seed_result["noise_scales"]) == {
            "thermal_hot",
            "fatigue_warm",
            "fatigue_cool",
            "fatigue_low_oxygen",
            "validation_hot_fatigue",
            "validation_cool_fatigue",
        }
        assert all(
            set(condition_scales) == set(OBSERVED_FIELDS)
            for condition_scales in seed_result["noise_scales"].values()
        )
