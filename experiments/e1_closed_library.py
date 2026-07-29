"""Run the reproducible E1 closed-library weak-discovery experiment."""

from pathlib import Path
import json
import sys
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wphid.pipeline import discover_closed
from wphid.truth.aging_ode import simulate_named_condition
from wphid.types import Candidate


SEED = 20260729
ARRHENIUS_TEMPERATURE = 1800.0
CONDITION_IDS = ("thermal_hold", "fatigue_rich", "oxygen_rich")
TRAIN_IDS = ("thermal_hold", "fatigue_rich")
VALIDATION_IDS = ("oxygen_rich",)
THRESHOLDS = np.geomspace(1.0e-4, 0.5, 48)
DEFAULT_OUTPUT_DIR = Path("artifacts/e1_closed_library")


def _arrhenius(states):
    return np.exp(-ARRHENIUS_TEMPERATURE / states["T"])


def closed_candidate_library() -> list[Candidate]:
    """Return the declared closed E1 library, including three distractors."""
    return [
        Candidate(
            "arr(T)*pO2",
            lambda states: _arrhenius(states) * states["pO2"],
            (1, -1, -2, 0),
            "G2",
        ),
        Candidate(
            "arr(T)*pO2*sigma",
            lambda states: _arrhenius(states) * states["pO2"] * states["sigma"],
            (2, -2, -4, 0),
            "G3",
        ),
        Candidate(
            "arr(T)*pO2*Df",
            lambda states: _arrhenius(states) * states["pO2"] * states["Df"],
            (1, -1, -2, 0),
            "G3",
        ),
        Candidate(
            "pO2",
            lambda states: states["pO2"],
            (1, -1, -2, 0),
            "F",
        ),
        Candidate(
            "pO2*sigma",
            lambda states: states["pO2"] * states["sigma"],
            (2, -2, -4, 0),
            "G2",
        ),
        Candidate(
            "arr(T)*pO2*sat(Df)",
            lambda states: _arrhenius(states)
            * states["pO2"]
            / (1.0 - states["Df"]),
            (1, -1, -2, 0),
            "G3",
        ),
    ]


def run_experiment(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Run E1 and write its compact, deterministic metrics artifact."""
    candidates = closed_candidate_library()
    trajectories = [simulate_named_condition(name) for name in CONDITION_IDS]
    result = discover_closed(
        trajectories=trajectories,
        candidates=candidates,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    support_names = [result.names[index] for index in result.support]
    metrics: dict[str, Any] = {
        "experiment": "e1_closed_library",
        "seed": SEED,
        "configuration": {
            "arrhenius_temperature": ARRHENIUS_TEMPERATURE,
            "candidate_names": list(result.names),
            "condition_ids": list(CONDITION_IDS),
            "noise_fraction": 0.0,
            "thresholds": [float(value) for value in THRESHOLDS],
            "train_ids": list(TRAIN_IDS),
            "validation_ids": list(VALIDATION_IDS),
            "weak_stride": int(result.diagnostics["weak_stride"]),
            "weak_window": int(result.diagnostics["weak_window"]),
        },
        "support": support_names,
        "coefficients": {
            name: float(coefficient)
            for name, coefficient in zip(
                result.names, result.coefficients, strict=True
            )
        },
        "validation_normalized_rmse": float(result.validation_error),
        "selected_threshold": float(result.diagnostics["selected_threshold"]),
        "selection_score": float(result.diagnostics["selection_score"]),
        "normalized_rmse_by_condition": {
            name: float(error)
            for name, error in result.diagnostics[
                "normalized_rmse_by_condition"
            ].items()
        },
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    result_metrics = run_experiment()
    print(
        "E1 support:",
        ", ".join(result_metrics["support"]),
        "validation normalized RMSE:",
        f"{result_metrics['validation_normalized_rmse']:.6g}",
    )
