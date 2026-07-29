import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from experiments.e3_missing_interaction import (
    EXPECTED_MISSING_INTERACTION,
    SEEDS,
    run_experiment,
)
from experiments.e3_missing_interaction_core import (
    THRESHOLDS,
    TRAIN_IDS,
    VALIDATION_IDS,
    _candidate_libraries,
    _observations,
)
from wphid.discovery.residual import discover_with_expansion
from wphid.pipeline import _condition_matrix
from wphid.sparse.stlsq import fit_stlsq
from wphid.types import Trajectory


TASK6_E1_VALIDATION_NRMSE = 0.00025879605792407254


@pytest.fixture(scope="module")
def e3_metrics(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("e3_missing_interaction")
    metrics = run_experiment(output_dir=output_dir)
    persisted = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    per_seed_dir = output_dir / "per_seed"
    persisted_per_seed = [
        json.loads(
            (per_seed_dir / f"seed_{seed}.json").read_text(encoding="utf-8")
        )
        for seed in SEEDS
    ]
    return metrics, persisted, persisted_per_seed, output_dir


def test_e3_discovers_missing_canonical_interaction_across_five_seeds(e3_metrics):
    """Training-only residual ranking plus validation admission must pass E3."""
    metrics, _, _, _ = e3_metrics

    assert metrics["aggregate"]["seeds"] == list(SEEDS)
    assert [result["seed"] for result in metrics["per_seed"]] == list(SEEDS)
    for result in metrics["per_seed"]:
        same_seed_closed_error = result[
            "same_seed_closed_library_diagnostic"
        ]["validation_normalized_rmse"]
        fixed_error = result["fixed_library"]["validation_normalized_rmse"]
        expanded_error = result["expanded_library"][
            "validation_normalized_rmse"
        ]
        task6_e1_error = result["task6_e1_reference"][
            "validation_normalized_rmse"
        ]
        admitted = result["expansion"]["admitted_candidates"]

        assert fixed_error >= 1.5 * same_seed_closed_error
        assert result["expansion"]["rounds"]
        first_round = result["expansion"]["rounds"][0]
        assert first_round["residual_ranking"]
        assert first_round["residual_base_fit"]["selection_data"] == (
            "training_conditions_only"
        )
        for evidence in first_round["residual_ranking"]:
            assert set(evidence["correlations_by_condition"]) == set(
                result["configuration"]["train_ids"]
            )
            assert {
                "median_absolute_correlation",
                "sign_consistency",
                "stable",
                "residual_score",
                "validation_error_reduction",
            } <= evidence.keys()
        assert set(result["configuration"]["validation_ids"]).isdisjoint(
            first_round["residual_condition_ids"]
        )
        assert EXPECTED_MISSING_INTERACTION in admitted
        assert expanded_error <= 1.2 * task6_e1_error
        assert len(
            set(admitted).difference({EXPECTED_MISSING_INTERACTION})
        ) <= 1
        assert all(result["gates"].values())

    assert metrics["aggregate"]["all_seeds_pass"]
    assert metrics["aggregate"]["passing_seed_count"] == len(SEEDS)


def test_expanded_error_gate_uses_fixed_task6_e1_reference(e3_metrics):
    """Replacing Task6 E1 with a noisy same-seed oracle breaks E3 reporting."""
    metrics, persisted, _, _ = e3_metrics

    assert persisted == metrics
    for result in persisted["per_seed"]:
        expanded_error = result["expanded_library"][
            "validation_normalized_rmse"
        ]
        task6_e1 = result["task6_e1_reference"]
        persisted_ratio = result["expanded_library"][
            "validation_nrmse_over_task6_e1"
        ]

        assert "closed_library" not in result
        assert task6_e1 == {
            "source": "Task6 E1",
            "validation_normalized_rmse": TASK6_E1_VALIDATION_NRMSE,
        }
        assert persisted_ratio == pytest.approx(
            expanded_error / TASK6_E1_VALIDATION_NRMSE
        )
        assert persisted_ratio <= 1.2
        assert result["gates"][
            "expanded_error_within_20_percent_of_task6_e1"
        ]


def test_e3_writes_lf_json_and_describes_target_only_do_noise(e3_metrics):
    """Platform newlines or ambiguous noise scope make byte hashes misleading."""
    metrics, persisted, persisted_per_seed, output_dir = e3_metrics

    assert persisted == metrics
    assert persisted_per_seed == metrics["per_seed"]
    artifact_paths = [
        output_dir / "metrics.json",
        *[
            output_dir / "per_seed" / f"seed_{seed}.json"
            for seed in SEEDS
        ],
    ]
    for artifact_path in artifact_paths:
        artifact_bytes = artifact_path.read_bytes()
        assert artifact_bytes.endswith(b"\n")
        assert b"\r\n" not in artifact_bytes
    for result in persisted["per_seed"]:
        assert result["configuration"]["observation_noise"] == {
            "applied_to": "Do_observation_only",
            "description": (
                "target-only Do observation noise; "
                "t, T, pO2, sigma, and Df are unchanged"
            ),
            "distribution": (
                "independent zero-mean Gaussian per time sample"
            ),
            "noise_fraction": 1.0e-4,
            "scale_per_condition": (
                "noise_fraction * max(abs(clean Do))"
            ),
        }
        assert result["configuration"]["json_newline"] == "LF"


def test_validation_target_cannot_change_training_residual_ranking():
    """Held-out Do may affect admission, never residual scores or stability."""
    trajectories = _observations(SEEDS[0])
    _, base_candidates, expansion_candidates = _candidate_libraries()
    training_matrices = [
        _condition_matrix(
            next(
                trajectory
                for trajectory in trajectories
                if trajectory.condition_id == condition_id
            ),
            base_candidates,
        )
        for condition_id in TRAIN_IDS
    ]
    low_threshold_fit = fit_stlsq(
        np.vstack([matrix for matrix, _ in training_matrices]),
        np.concatenate([target for _, target in training_matrices]),
        float(THRESHOLDS[0]),
    )
    validation = next(
        trajectory
        for trajectory in trajectories
        if trajectory.condition_id in VALIDATION_IDS
    )
    validation_features = np.column_stack(
        [
            candidate.evaluate(validation.states)
            for candidate in base_candidates
        ]
    )
    validation_only_target = cumulative_trapezoid(
        validation_features @ low_threshold_fit.coefficients,
        validation.t,
        initial=0.0,
    )
    perturbed = [
        Trajectory(
            t=trajectory.t,
            states={
                **dict(trajectory.states),
                "Do": (
                    validation_only_target
                    if trajectory.condition_id in VALIDATION_IDS
                    else trajectory.states["Do"]
                ),
            },
            condition_id=trajectory.condition_id,
        )
        for trajectory in trajectories
    ]

    baseline = discover_with_expansion(
        trajectories,
        base_candidates,
        expansion_candidates,
        TRAIN_IDS,
        VALIDATION_IDS,
        THRESHOLDS,
    )
    changed_validation = discover_with_expansion(
        perturbed,
        base_candidates,
        expansion_candidates,
        TRAIN_IDS,
        VALIDATION_IDS,
        THRESHOLDS,
    )

    baseline_ranking = baseline.diagnostics["expansion_rounds"][0][
        "residual_ranking"
    ]
    changed_ranking = changed_validation.diagnostics["expansion_rounds"][0][
        "residual_ranking"
    ]
    evidence_fields = (
        "candidate",
        "correlations_by_condition",
        "median_absolute_correlation",
        "sign_consistency",
        "stable",
        "residual_score",
    )
    assert [
        {field: evidence[field] for field in evidence_fields}
        for evidence in changed_ranking
    ] == [
        {field: evidence[field] for field in evidence_fields}
        for evidence in baseline_ranking
    ]


def test_e3_script_runs_from_outside_repository(tmp_path):
    """The exact gate entry point must bootstrap both project and source paths."""
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
    assert "E3 seeds passing: 5/5" in completed.stdout
    persisted = json.loads(
        (
            tmp_path
            / "artifacts"
            / "e3_missing_interaction"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["aggregate"]["all_seeds_pass"]
