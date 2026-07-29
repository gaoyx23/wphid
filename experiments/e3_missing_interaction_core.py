"""Run the reproducible E3 residual-driven missing-interaction gate."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

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
TASK6_E1_VALIDATION_NRMSE = 0.00025879605792407254
DEFAULT_OUTPUT_DIR = Path("artifacts/e3_missing_interaction")
_JSON_NEWLINE = "LF"
_OBSERVATION_NOISE = {
    "applied_to": "Do_observation_only",
    "description": (
        "target-only Do observation noise; "
        "t, T, pO2, sigma, and Df are unchanged"
    ),
    "distribution": "independent zero-mean Gaussian per time sample",
    "noise_fraction": NOISE_FRACTION,
    "scale_per_condition": "noise_fraction * max(abs(clean Do))",
}


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
            "Do": oxidation + rng.normal(
                scale=noise_scale, size=oxidation.size
            ),
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
    """Run one target-noise E3 replicate with truth isolated from discovery."""
    trajectories = _observations(seed)
    closed_candidates, fixed_candidates, expansion_candidates = (
        _candidate_libraries()
    )
    same_seed_closed_result = discover_closed(
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
        and rounds[0]["residual_base_fit"]["selection_data"]
        == "training_conditions_only"
        and all(
            set(item["correlations_by_condition"]) == set(TRAIN_IDS)
            for item in rounds[0]["residual_ranking"]
        )
    )
    expanded_summary = _result_summary(expanded_result)
    expanded_summary["validation_nrmse_over_task6_e1"] = (
        expanded_result.validation_error / TASK6_E1_VALIDATION_NRMSE
    )
    same_seed_closed_summary = _result_summary(same_seed_closed_result)
    same_seed_closed_summary["role"] = (
        "noisy same-seed closed-library diagnostic; not the Task6 E1 reference"
    )
    gates = {
        "fixed_error_at_least_50_percent_above_same_seed_closed_diagnostic": (
            fixed_result.validation_error
            >= 1.5 * same_seed_closed_result.validation_error
        ),
        "structured_cross_condition_residual_evidence": structured_evidence,
        "missing_interaction_admitted": (
            EXPECTED_MISSING_INTERACTION in admitted
        ),
        "expanded_error_within_20_percent_of_task6_e1": (
            expanded_result.validation_error
            <= 1.2 * TASK6_E1_VALIDATION_NRMSE
        ),
        "at_most_one_false_interaction": len(false_interactions) <= 1,
    }
    return {
        "seed": int(seed),
        "configuration": {
            "arrhenius_temperature": ARRHENIUS_TEMPERATURE,
            "observation_noise": dict(_OBSERVATION_NOISE),
            "json_newline": _JSON_NEWLINE,
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
        "task6_e1_reference": {
            "source": "Task6 E1",
            "validation_normalized_rmse": TASK6_E1_VALIDATION_NRMSE,
        },
        "same_seed_closed_library_diagnostic": same_seed_closed_summary,
        "fixed_library": _result_summary(fixed_result),
        "expanded_library": expanded_summary,
        "expansion": {
            "admitted_candidates": admitted,
            "false_interactions": false_interactions,
            "rounds": rounds,
        },
        "gates": gates,
    }


def _json_bytes(value: Any) -> bytes:
    """Serialize deterministic UTF-8 JSON with exactly one trailing LF."""
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def run_experiment(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Run all E3 replicates and persist LF-normalized JSON metrics."""
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
            "task6_e1_validation_normalized_rmse": (
                TASK6_E1_VALIDATION_NRMSE
            ),
            "maximum_expanded_nrmse_over_task6_e1": max(
                result["expanded_library"][
                    "validation_nrmse_over_task6_e1"
                ]
                for result in per_seed
            ),
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
        seed_path.write_bytes(_json_bytes(result))
    (destination / "metrics.json").write_bytes(_json_bytes(metrics))
    return metrics
