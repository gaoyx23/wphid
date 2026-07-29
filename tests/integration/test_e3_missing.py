import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.e3_missing_interaction import (
    EXPECTED_MISSING_INTERACTION,
    SEEDS,
    run_experiment,
)


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
    return metrics, persisted, persisted_per_seed


def test_e3_discovers_missing_canonical_interaction_across_five_seeds(e3_metrics):
    """Training-only residual ranking plus validation admission must pass E3."""
    metrics, _, _ = e3_metrics

    assert metrics["aggregate"]["seeds"] == list(SEEDS)
    assert [result["seed"] for result in metrics["per_seed"]] == list(SEEDS)
    for result in metrics["per_seed"]:
        closed_error = result["closed_library"]["validation_normalized_rmse"]
        fixed_error = result["fixed_library"]["validation_normalized_rmse"]
        expanded_error = result["expanded_library"][
            "validation_normalized_rmse"
        ]
        admitted = result["expansion"]["admitted_candidates"]

        assert fixed_error >= 1.5 * closed_error
        assert result["expansion"]["rounds"]
        first_round = result["expansion"]["rounds"][0]
        assert first_round["residual_ranking"]
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
        assert expanded_error <= 1.2 * closed_error
        assert len(
            set(admitted).difference({EXPECTED_MISSING_INTERACTION})
        ) <= 1
        assert all(result["gates"].values())

    assert metrics["aggregate"]["all_seeds_pass"]
    assert metrics["aggregate"]["passing_seed_count"] == len(SEEDS)


def test_e3_writes_exact_per_seed_and_aggregate_metrics(e3_metrics):
    """Dropping a seed artifact or changing persisted values breaks the gate."""
    metrics, persisted, persisted_per_seed = e3_metrics

    assert persisted == metrics
    assert persisted_per_seed == metrics["per_seed"]


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
