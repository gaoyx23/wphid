"""Run the reproducible E3 residual-driven missing-interaction gate."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.e1_closed_library import (
    ARRHENIUS_TEMPERATURE,
    THRESHOLDS,
    closed_candidate_library,
)
from wphid.discovery.residual import discover_with_expansion
from wphid.grammar.atoms import PRESSURE, aging_atoms
from wphid.grammar.generate import generate_candidates
from wphid.pipeline import discover_closed
from wphid.truth.aging_ode import simulate_condition
from wphid.types import DiscoveryResult, Trajectory


SEEDS = (11, 23, 47, 89, 131)
NOISE_FRACTION = 1.0e-4
TRAIN_IDS = ("fatigue_warm", "fatigue_oxygen")
VALIDATION_IDS = ("fatigue_validation",)
EXPECTED_MISSING_INTERACTION = "arr(T;theta=1800)*pO2*Df"
DEFAULT_OUTPUT_DIR = Path("artifacts/e3_missing_interaction")


def _forcing_profiles(t: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    phase = 2.0 * np.pi * t / t[-1]
    return {
        "fatigue_warm": {
            "T": 380.0 + 9.0 * np.sin(phase),
            "pO2": 0.28 + 0.06 * np.cos(2.0 * phase),
            "sigma": 55.0 + 35.0 * np.sin(5.0 * phase + 0.2),
        },
        "fatigue_oxygen": {
            "T": 348.0 + 12.0 * np.cos(phase),
            "pO2": 0.58 + 0.09 * np.sin(3.0 * phase + 0.4),
            "sigma": 48.0 + 40.0 * np.sin(4.0 * phase + 0.5),
        },
        "fatigue_validation": {
            "T": 364.0 + 11.0 * np.sin(2.0 * phase + 0.3),
            "pO2": 0.43 + 0.10 * np.cos(3.0 * phase + 0.7),
            "sigma": 52.0 + 38.0 * np.sin(6.0 * phase + 0.8),
        },
    }


def _observations(seed: int) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 12.0, 241)
    trajectories = []
    for condition_id, forcing in _forcing_profiles(time).items():
        clean = simulate_condition(condition_id, time, forcing)
        oxidation = np.asarray(clean.states["Do"])
        noise_scale = NOISE_FRACTION * float(np.max(np.abs(oxidation)))
        noisy_states = {
            **clean.states,
            "Do": oxidation + rng.normal(scale=noise_scale, size=oxidation.size),
        }
        trajectories.append(
            Trajectory(
                t=clean.t,
                states=noisy_states,
                condition_id=condition_id,
            )
        )
    return trajectories


def _candidate_libraries():
    closed = closed_candidate_library()
    fixed = [
        candidate
        for candidate in closed
        if candidate.name != EXPECTED_MISSING_INTERACTION
    ]
    generated = generate_candidates(
        aging_atoms(arrhenius_temperature=ARRHENIUS_TEMPERATURE),
        PRESSURE,
    )
    fixed_names = {candidate.name for candidate in fixed}
    expansion = [
        candidate
        for candidate in generated
        if candidate.group == "G3" and candidate.name not in fixed_names
    ]
    return closed, fixed, expansion


def _support_names(result: DiscoveryResult) -> list[str]:
    return [result.names[index] for index in result.support]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _result_summary(result: DiscoveryResult) -> dict[str, Any]:
    return {
        "candidate_names": list(result.names),
        "support": _support_names(result),
        "coefficients": {
            name: float(coefficient)
            for name, coefficient in zip(
                result.names, result.coefficients, strict=True
            )
        },
        "validation_normalized_rmse": float(result.validation_error),
    }


def run_seed(seed: int) -> dict[str, Any]:
    """Run one noisy E3 replicate without exposing truth support to discovery."""
    trajectories = _observations(seed)
    closed_candidates, fixed_candidates, expansion_candidates = (
        _candidate_libraries()
    )
    closed_result = discover_closed(
        trajectories=trajectories,
        candidates=closed_candidates,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    fixed_result = discover_closed(
        trajectories=trajectories,
        candidates=fixed_candidates,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    expanded_result = discover_with_expansion(
        trajectories=trajectories,
        base_candidates=fixed_candidates,
        expansion_candidates=expansion_candidates,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
        max_rounds=2,
    )
    admitted = list(expanded_result.diagnostics["admitted_expansions"])
    rounds = _jsonable(expanded_result.diagnostics["expansion_rounds"])
    false_interactions = sorted(
        set(admitted).difference({EXPECTED_MISSING_INTERACTION})
    )
    structured_evidence = bool(
        rounds
        and rounds[0]["residual_ranking"]
        and all(
            set(item["correlations_by_condition"]) == set(TRAIN_IDS)
            for item in rounds[0]["residual_ranking"]
        )
    )
    gates = {
        "fixed_error_at_least_50_percent_above_closed": (
            fixed_result.validation_error >= 1.5 * closed_result.validation_error
        ),
        "structured_cross_condition_residual_evidence": structured_evidence,
        "missing_interaction_admitted": (
            EXPECTED_MISSING_INTERACTION in admitted
        ),
        "expanded_error_within_20_percent_of_closed": (
            expanded_result.validation_error
            <= 1.2 * closed_result.validation_error
        ),
        "at_most_one_false_interaction": len(false_interactions) <= 1,
    }
    return {
        "seed": int(seed),
        "configuration": {
            "arrhenius_temperature": ARRHENIUS_TEMPERATURE,
            "noise_fraction": NOISE_FRACTION,
            "train_ids": list(TRAIN_IDS),
            "validation_ids": list(VALIDATION_IDS),
            "thresholds": [float(value) for value in THRESHOLDS],
            "base_candidate_names": [
                candidate.name for candidate in fixed_candidates
            ],
            "expansion_candidate_names": [
                candidate.name for candidate in expansion_candidates
            ],
        },
        "closed_library": _result_summary(closed_result),
        "fixed_library": _result_summary(fixed_result),
        "expanded_library": _result_summary(expanded_result),
        "expansion": {
            "admitted_candidates": admitted,
            "false_interactions": false_interactions,
            "rounds": rounds,
        },
        "gates": gates,
    }


def run_experiment(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Run all E3 replicates and persist per-seed plus aggregate metrics."""
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must be non-empty")
    per_seed = [run_seed(seed) for seed in seed_values]
    passing_seed_count = sum(
        all(result["gates"].values()) for result in per_seed
    )
    metrics = {
        "experiment": "e3_missing_interaction",
        "per_seed": per_seed,
        "aggregate": {
            "seeds": list(seed_values),
            "passing_seed_count": passing_seed_count,
            "all_seeds_pass": passing_seed_count == len(seed_values),
            "missing_interaction_detection_rate": (
                sum(
                    result["gates"]["missing_interaction_admitted"]
                    for result in per_seed
                )
                / len(seed_values)
            ),
            "maximum_false_interactions": max(
                len(result["expansion"]["false_interactions"])
                for result in per_seed
            ),
        },
    }

    destination = Path(output_dir)
    per_seed_destination = destination / "per_seed"
    per_seed_destination.mkdir(parents=True, exist_ok=True)
    for result in per_seed:
        seed_path = per_seed_destination / f"seed_{result['seed']}.json"
        seed_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    experiment_metrics = run_experiment()
    aggregate = experiment_metrics["aggregate"]
    print(
        "E3 seeds passing:",
        f"{aggregate['passing_seed_count']}/{len(aggregate['seeds'])}",
        "detection rate:",
        f"{aggregate['missing_interaction_detection_rate']:.3f}",
        "max false interactions:",
        aggregate["maximum_false_interactions"],
    )
