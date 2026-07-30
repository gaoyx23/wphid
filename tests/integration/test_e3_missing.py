import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.e3_missing_interaction import (
    DEFAULT_E1_METRICS_PATH,
    EXPECTED_MISSING_INTERACTION,
    SEEDS,
    TEST_IDS,
    run_experiment,
    run_seed,
)
from experiments.e3_missing_interaction_core import (
    TRAIN_IDS,
    VALIDATION_IDS,
    _observations,
)
from wphid.types import Trajectory


@pytest.fixture(scope="module")
def e3_metrics(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("e3_missing_interaction")
    metrics = run_experiment(
        output_dir=output_dir,
        e1_metrics_path=DEFAULT_E1_METRICS_PATH,
    )
    persisted = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    return metrics, persisted, output_dir


def test_e3_persists_three_stage_finite_pool_evidence(e3_metrics):
    """Missing stage labels or quantitative evidence would overstate E3."""
    metrics, persisted, _ = e3_metrics

    assert persisted == metrics
    assert metrics["aggregate"]["seeds"] == list(SEEDS)
    assert metrics["scope"]["admission"] == "finite-pool validation admission"
    assert metrics["scope"]["claim_boundary"] == (
        "controlled finite-pool evidence; not unrestricted grammar discovery"
    )
    for result in metrics["per_seed"]:
        roles = result["configuration"]["condition_roles"]
        assert roles == {
            "residual_screening_and_training": list(TRAIN_IDS),
            "finite_pool_validation_admission": list(VALIDATION_IDS),
            "independent_test_evaluation": list(TEST_IDS),
        }
        assert result["parameter_selection"]["selection_data"] == (
            "training_conditions_only"
        )
        assert result["expansion"]["admission_name"] == (
            "finite-pool validation admission"
        )
        assert result["expansion"]["finite_pool"]
        assert set(result["expansion"]["admitted_candidates"]).issubset(
            set(result["expansion"]["finite_pool"])
        )
        assert result["expansion"]["rounds"]
        for round_evidence in result["expansion"]["rounds"]:
            assert round_evidence["stage_name"] == (
                "training-only residual screening"
            )
            assert round_evidence["residual_condition_ids"] == list(TRAIN_IDS)
            assert round_evidence["stability_thresholds"] == {
                "minimum_absolute_correlation": 0.25,
                "minimum_sign_consistency": 0.75,
                "minimum_supporting_condition_count": 2,
            }
            assert round_evidence["validation_admission"][
                "stage_name"
            ] == "finite-pool validation admission"
            assert round_evidence["validation_admission"][
                "condition_ids"
            ] == list(VALIDATION_IDS)
            for score in round_evidence["residual_ranking"]:
                assert {
                    "correlations_by_condition",
                    "median_absolute_correlation",
                    "sign_consistency",
                    "stable",
                    "residual_score",
                    "validation_error_reduction",
                } <= score.keys()
                assert set(score["correlations_by_condition"]) == set(
                    TRAIN_IDS
                )

        for model_name in (
            "same_seed_closed_library_diagnostic",
            "fixed_library",
            "expanded_library",
        ):
            model = result[model_name]
            assert np.isfinite(model["validation_normalized_rmse"])
            assert np.isfinite(model["test_normalized_rmse"])
        assert result["expanded_library"]["test_normalized_rmse"] == pytest.approx(
            result["independent_test_evaluation"][
                "expanded_library_normalized_rmse"
            ]
        )
    detection_count = sum(
        EXPECTED_MISSING_INTERACTION
        in result["expansion"]["admitted_candidates"]
        for result in metrics["per_seed"]
    )
    passing_seed_count = sum(
        all(result["gates"].values()) for result in metrics["per_seed"]
    )
    assert metrics["aggregate"]["missing_interaction_detection_rate"] == (
        detection_count / len(SEEDS)
    )
    assert metrics["aggregate"]["passing_seed_count"] == passing_seed_count
    assert metrics["aggregate"]["e3_gate_pass"] == (
        passing_seed_count == len(SEEDS)
    )
    assert metrics["aggregate"]["overall_phase_decision"] == "NO-GO"


def test_e3_loads_e1_reference_and_cannot_override_e1_no_go(tmp_path):
    """Changing the supplied E1 artifact must change E3's persisted reference."""
    reference_nrmse = 0.375
    e1_path = tmp_path / "e1-metrics.json"
    e1_path.write_text(
        json.dumps(
            {
                "experiment": "e1_closed_library",
                "aggregate": {
                    "all_seeds_pass": False,
                    "gate": {
                        "metric": "minimum_per_seed_support_f1",
                        "threshold": 0.9,
                        "observed": 0.75,
                        "pass": False,
                    },
                    "reference_validation_normalized_rmse": reference_nrmse,
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = run_experiment(
        output_dir=tmp_path / "e3",
        seeds=(SEEDS[0],),
        e1_metrics_path=e1_path,
    )

    assert metrics["e1_reference"][
        "reference_validation_normalized_rmse"
    ] == reference_nrmse
    assert not metrics["aggregate"]["e1_gate_pass"]
    assert metrics["aggregate"]["overall_phase_decision"] == "NO-GO"
    assert metrics["aggregate"]["overall_phase_decision_rule"] == (
        "E1 and E3 must both pass; E3 cannot override E1"
    )
    ratio = metrics["per_seed"][0]["expanded_library"][
        "test_nrmse_over_e1_reference"
    ]
    assert ratio == pytest.approx(
        metrics["per_seed"][0]["expanded_library"]["test_normalized_rmse"]
        / reference_nrmse
    )


def _replace_target(
    trajectories: list[Trajectory],
    condition_ids: tuple[str, ...],
    transform,
) -> list[Trajectory]:
    return [
        Trajectory(
            t=trajectory.t,
            states={
                **dict(trajectory.states),
                "Do": (
                    transform(trajectory)
                    if trajectory.condition_id in condition_ids
                    else trajectory.states["Do"]
                ),
            },
            condition_id=trajectory.condition_id,
        )
        for trajectory in trajectories
    ]


def test_test_target_cannot_change_theta_threshold_ranking_or_admission():
    """The independent test target may change only post-selection evaluation."""
    trajectories = _observations(SEEDS[0])
    perturbed = _replace_target(
        trajectories,
        TEST_IDS,
        lambda trajectory: (
            7.0 * np.asarray(trajectory.states["Do"])
            + 0.2 * trajectory.t**2
        ),
    )

    baseline = run_seed(SEEDS[0], trajectories=trajectories)
    changed = run_seed(SEEDS[0], trajectories=perturbed)

    assert changed["parameter_selection"] == baseline["parameter_selection"]
    assert changed["fixed_library"]["selected_threshold"] == baseline[
        "fixed_library"
    ]["selected_threshold"]
    assert changed["same_seed_closed_library_diagnostic"][
        "selected_threshold"
    ] == baseline["same_seed_closed_library_diagnostic"]["selected_threshold"]
    assert changed["expanded_library"]["selected_threshold"] == baseline[
        "expanded_library"
    ]["selected_threshold"]
    assert changed["expansion"] == baseline["expansion"]
    assert changed["fixed_library"]["validation_normalized_rmse"] == baseline[
        "fixed_library"
    ]["validation_normalized_rmse"]
    assert changed["expanded_library"]["validation_normalized_rmse"] == baseline[
        "expanded_library"
    ]["validation_normalized_rmse"]
    assert changed["expanded_library"]["test_normalized_rmse"] != baseline[
        "expanded_library"
    ]["test_normalized_rmse"]


def test_validation_target_cannot_change_theta_or_training_residual_screening():
    """Validation may admit a candidate but cannot alter train-only evidence."""
    trajectories = _observations(SEEDS[0])
    perturbed = _replace_target(
        trajectories,
        VALIDATION_IDS,
        lambda trajectory: np.asarray(trajectory.states["Do"])[::-1],
    )

    baseline = run_seed(SEEDS[0], trajectories=trajectories)
    changed = run_seed(SEEDS[0], trajectories=perturbed)

    assert changed["parameter_selection"] == baseline["parameter_selection"]
    evidence_fields = (
        "candidate",
        "correlations_by_condition",
        "median_absolute_correlation",
        "sign_consistency",
        "stable",
        "residual_score",
    )
    for changed_round, baseline_round in zip(
        changed["expansion"]["rounds"],
        baseline["expansion"]["rounds"],
        strict=True,
    ):
        assert [
            {field: score[field] for field in evidence_fields}
            for score in changed_round["residual_ranking"]
        ] == [
            {field: score[field] for field in evidence_fields}
            for score in baseline_round["residual_ranking"]
        ]


def test_e3_writes_lf_compact_json(e3_metrics):
    """Committed evidence stays deterministic and avoids redundant seed files."""
    metrics, persisted, output_dir = e3_metrics

    assert persisted == metrics
    artifact_bytes = (output_dir / "metrics.json").read_bytes()
    assert artifact_bytes.endswith(b"\n")
    assert b"\r\n" not in artifact_bytes
    assert not (output_dir / "per_seed").exists()


def test_e3_script_runs_from_outside_repository(tmp_path):
    """The gate entry point resolves its committed E1 artifact explicitly."""
    script = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "e3_missing_interaction.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "E3 seeds passing:" in completed.stdout
    assert "overall phase: NO-GO" in completed.stdout
    persisted = json.loads(
        (
            tmp_path
            / "artifacts"
            / "e3_missing_interaction"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["aggregate"]["overall_phase_decision"] == "NO-GO"
