"""Run the reproducible noisy E1 closed-library discovery gate."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wphid.dimensions import DAMAGE_RATE, subtract_dimensions
from wphid.grammar.atoms import PRESSURE, aging_atoms
from wphid.grammar.generate import generate_candidates
from wphid.pipeline import discover_profiled_closed
from wphid.truth.aging_ode import (
    oxidation_evaluation_truth,
    simulate_condition,
)
from wphid.types import Candidate, DiscoveryResult, Trajectory


PROVENANCE_ID = 20260729
NOISE_SEEDS = (101, 203, 307, 401, 509)
NOISE_FRACTION = 0.05
SUPPORT_F1_THRESHOLD = 0.90
OBSERVED_FIELDS = ("T", "pO2", "sigma", "Df", "Do")
THETA_GRID = (1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0)
CONDITION_IDS = (
    "thermal_hot",
    "fatigue_warm",
    "fatigue_cool",
    "fatigue_low_oxygen",
    "validation_hot_fatigue",
    "validation_cool_fatigue",
)
TRAIN_IDS = CONDITION_IDS[:4]
VALIDATION_IDS = CONDITION_IDS[4:]
THRESHOLDS = np.geomspace(1.0e-4, 0.5, 48)
DEFAULT_OUTPUT_DIR = Path("artifacts/e1_closed_library")
PRESSURE_SQUARED = (2, -2, -4, 0)
_OBSERVATION_NOISE = {
    "applied_to": list(OBSERVED_FIELDS),
    "distribution": "independent zero-mean Gaussian per field and time sample",
    "noise_fraction": NOISE_FRACTION,
    "scale_per_condition_field": "noise_fraction * RMS(clean field)",
}


def closed_candidate_library(
    arrhenius_temperature: float,
) -> list[Candidate]:
    """Return the shared grammar's closed E1 library and three distractors."""
    atoms = aging_atoms(arrhenius_temperature=arrhenius_temperature)
    arrhenius_name = next(
        candidate.name
        for candidate in atoms
        if candidate.name.startswith("arr(T;")
    )
    generated = generate_candidates(
        atoms,
        PRESSURE,
        coefficient_dimension=subtract_dimensions(DAMAGE_RATE, PRESSURE),
    )
    generated += generate_candidates(
        atoms,
        PRESSURE_SQUARED,
        coefficient_dimension=subtract_dimensions(
            DAMAGE_RATE, PRESSURE_SQUARED
        ),
    )
    by_name = {candidate.name: candidate for candidate in generated}
    names = (
        f"{arrhenius_name}*pO2",
        f"{arrhenius_name}*pO2*sigma",
        f"{arrhenius_name}*pO2*Df",
        "pO2",
        "pO2*sigma",
    )
    return [by_name[name] for name in names]


def candidate_profiles(
    theta_values: Sequence[float] = THETA_GRID,
) -> dict[float, list[Candidate]]:
    """Build a declared theta grid without consulting truth or held-out data."""
    profiles = {}
    for raw_theta in theta_values:
        theta = float(raw_theta)
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError("theta grid values must be finite and positive")
        if theta in profiles:
            raise ValueError("theta grid values must be unique")
        profiles[theta] = closed_candidate_library(theta)
    if not profiles:
        raise ValueError("theta grid must be non-empty")
    return profiles


def _clean_observations() -> tuple[Trajectory, ...]:
    time = np.linspace(0.0, 20.0, 401)
    phase = 2.0 * np.pi * time / time[-1]
    forcing_profiles = {
        "thermal_hot": {
            "T": 440.0 + 10.0 * np.sin(phase),
            "pO2": 0.22 + 0.04 * np.cos(2.0 * phase),
            "sigma": np.full_like(time, 105.0),
        },
        "fatigue_warm": {
            "T": 395.0 + 14.0 * np.sin(2.0 * phase),
            "pO2": 0.30 + 0.08 * np.cos(3.0 * phase),
            "sigma": 48.0 + 25.0 * np.sin(5.0 * phase + 0.3),
        },
        "fatigue_cool": {
            "T": 330.0 + 12.0 * np.cos(1.5 * phase),
            "pO2": 0.62 + 0.10 * np.sin(2.0 * phase + 0.5),
            "sigma": 52.0 + 30.0 * np.sin(4.0 * phase + 0.2),
        },
        "fatigue_low_oxygen": {
            "T": 370.0 + 18.0 * np.cos(2.5 * phase),
            "pO2": 0.15 + 0.04 * np.sin(3.0 * phase + 0.7),
            "sigma": 45.0 + 35.0 * np.sin(6.0 * phase + 0.4),
        },
        "validation_hot_fatigue": {
            "T": 420.0 + 16.0 * np.sin(1.5 * phase + 0.2),
            "pO2": 0.48 + 0.09 * np.cos(4.0 * phase + 0.6),
            "sigma": 50.0 + 28.0 * np.sin(5.0 * phase + 0.9),
        },
        "validation_cool_fatigue": {
            "T": 350.0 + 20.0 * np.cos(2.0 * phase + 0.4),
            "pO2": 0.75 + 0.08 * np.sin(2.5 * phase + 0.3),
            "sigma": 47.0 + 32.0 * np.sin(3.5 * phase + 0.8),
        },
    }
    if tuple(forcing_profiles) != CONDITION_IDS:
        raise RuntimeError("E1 condition ordering must match the declared split")
    return tuple(
        simulate_condition(condition_id, time, forcing)
        for condition_id, forcing in forcing_profiles.items()
    )


def _noisy_observations_with_scales(
    seed: int,
    clean_observations: Sequence[Trajectory],
) -> tuple[list[Trajectory], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(int(seed))
    observations = []
    scales = {}
    for clean in clean_observations:
        noisy_states = {}
        condition_scales = {}
        for name in OBSERVED_FIELDS:
            values = np.asarray(clean.states[name], dtype=float)
            rms = float(np.sqrt(np.mean(values**2)))
            noise_scale = NOISE_FRACTION * rms
            condition_scales[name] = noise_scale
            noisy_states[name] = values + rng.normal(
                scale=noise_scale,
                size=values.shape,
            )
        scales[clean.condition_id] = condition_scales
        observations.append(
            Trajectory(
                t=clean.t,
                states=noisy_states,
                condition_id=clean.condition_id,
            )
        )
    return observations, scales


def noisy_observations(seed: int) -> list[Trajectory]:
    """Return genuine multivariate 5% noisy observation trajectories."""
    observations, _ = _noisy_observations_with_scales(
        seed,
        _clean_observations(),
    )
    return observations


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


def _support_metrics(
    inferred: set[str],
    expected: set[str],
) -> dict[str, float | int]:
    true_positives = len(inferred & expected)
    false_positives = len(inferred - expected)
    false_negatives = len(expected - inferred)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positives / recall_denominator
        if recall_denominator
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluation_metrics(
    result: DiscoveryResult,
    seed: int,
    noise_scales: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Read hidden truth only after discovery and compute gate metrics."""
    truth = oxidation_evaluation_truth()
    selected_theta = float(result.diagnostics["selected_parameter"])
    true_theta = float(truth["arrhenius_theta"])
    truth_by_factor = {
        str(name): float(value)
        for name, value in truth["coefficient_by_factor"].items()
    }
    arrhenius_name = next(
        name.split("*", maxsplit=1)[0]
        for name in result.names
        if name.startswith("arr(T;theta=")
    )
    expected_coefficients = {
        f"{arrhenius_name}*{factor}": coefficient
        for factor, coefficient in truth_by_factor.items()
    }
    expected_support = set(expected_coefficients)
    inferred_support = {
        result.names[index] for index in result.support
    }
    support_metrics = _support_metrics(inferred_support, expected_support)
    coefficient_by_name = {
        name: float(coefficient)
        for name, coefficient in zip(
            result.names, result.coefficients, strict=True
        )
    }
    truth_vector = np.asarray(tuple(expected_coefficients.values()))
    estimate_vector = np.asarray(
        tuple(
            coefficient_by_name[name]
            for name in expected_coefficients
        )
    )
    coefficient_relative_l2_error = float(
        np.linalg.norm(estimate_vector - truth_vector)
        / np.linalg.norm(truth_vector)
    )
    relative_absolute_error_by_term = {
        name: abs(coefficient_by_name[name] - truth_coefficient)
        / abs(truth_coefficient)
        for name, truth_coefficient in expected_coefficients.items()
    }
    profile_summary = []
    selected_profile_identity = None
    for profile in result.diagnostics["parameter_profile_history"]:
        profile_record = {
            key: _jsonable(value)
            for key, value in profile.items()
            if key != "threshold_history"
        }
        profile_summary.append(profile_record)
        if float(profile["parameter"]) == selected_theta:
            selected_profile_identity = list(
                profile["candidate_identities"]
            )
    return {
        "seed": int(seed),
        "support": sorted(inferred_support),
        "expected_support": sorted(expected_support),
        "support_metrics": support_metrics,
        "support_gate_pass": (
            support_metrics["f1"] >= SUPPORT_F1_THRESHOLD
        ),
        "coefficients": coefficient_by_name,
        "coefficient_metrics": {
            "relative_l2_error": coefficient_relative_l2_error,
            "relative_absolute_error_by_term": (
                relative_absolute_error_by_term
            ),
        },
        "selected_theta": selected_theta,
        "selected_theta_candidate_identities": selected_profile_identity,
        "evaluation_true_theta": true_theta,
        "theta_relative_error": abs(selected_theta - true_theta) / true_theta,
        "parameter_selection_data": result.diagnostics[
            "parameter_selection_data"
        ],
        "parameter_selection_rule": result.diagnostics[
            "parameter_selection_rule"
        ],
        "parameter_profile_summary": profile_summary,
        "selected_threshold": float(
            result.diagnostics["selected_threshold"]
        ),
        "selection_score": float(result.diagnostics["selection_score"]),
        "validation_normalized_rmse": float(result.validation_error),
        "normalized_rmse_by_condition": {
            name: float(error)
            for name, error in result.diagnostics[
                "normalized_rmse_by_condition"
            ].items()
        },
        "noise_scales": _jsonable(noise_scales),
    }


def run_seed(
    seed: int,
    clean_observations: Sequence[Trajectory] | None = None,
    profiles: Mapping[float, Sequence[Candidate]] | None = None,
) -> dict[str, Any]:
    """Run one E1 replicate with truth absent from discovery inputs."""
    clean = (
        tuple(clean_observations)
        if clean_observations is not None
        else _clean_observations()
    )
    observations, noise_scales = _noisy_observations_with_scales(seed, clean)
    candidate_profile_values = (
        profiles if profiles is not None else candidate_profiles()
    )
    result = discover_profiled_closed(
        trajectories=observations,
        candidate_profiles=candidate_profile_values,
        parameter_name="arrhenius_theta",
        target_dimension=DAMAGE_RATE,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    return _evaluation_metrics(result, seed, noise_scales)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def run_experiment(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = NOISE_SEEDS,
) -> dict[str, Any]:
    """Run canonical E1 replicates and persist deterministic gate metrics."""
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must be non-empty")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    clean = _clean_observations()
    profiles = candidate_profiles()
    per_seed = [
        run_seed(seed, clean_observations=clean, profiles=profiles)
        for seed in seed_values
    ]
    minimum_support_f1 = min(
        result["support_metrics"]["f1"] for result in per_seed
    )
    passing_seed_count = sum(
        result["support_gate_pass"] for result in per_seed
    )
    metrics: dict[str, Any] = {
        "experiment": "e1_closed_library",
        "provenance_id": PROVENANCE_ID,
        "configuration": {
            "condition_ids": list(CONDITION_IDS),
            "train_ids": list(TRAIN_IDS),
            "validation_ids": list(VALIDATION_IDS),
            "theta_grid": list(THETA_GRID),
            "theta_selection_data": "training_conditions_only",
            "thresholds": [float(value) for value in THRESHOLDS],
            "observation_noise": dict(_OBSERVATION_NOISE),
            "target_dimension": list(DAMAGE_RATE),
            "normalized_rmse_definition": (
                "RMS(weak residual)/RMS(weak target)"
            ),
        },
        "per_seed": per_seed,
        "aggregate": {
            "seeds": list(seed_values),
            "support_f1_threshold": SUPPORT_F1_THRESHOLD,
            "minimum_support_f1": minimum_support_f1,
            "mean_support_precision": float(
                np.mean(
                    [
                        result["support_metrics"]["precision"]
                        for result in per_seed
                    ]
                )
            ),
            "mean_support_recall": float(
                np.mean(
                    [
                        result["support_metrics"]["recall"]
                        for result in per_seed
                    ]
                )
            ),
            "maximum_coefficient_relative_l2_error": max(
                result["coefficient_metrics"]["relative_l2_error"]
                for result in per_seed
            ),
            "maximum_theta_relative_error": max(
                result["theta_relative_error"] for result in per_seed
            ),
            "reference_validation_normalized_rmse": max(
                result["validation_normalized_rmse"]
                for result in per_seed
            ),
            "reference_validation_nrmse_definition": (
                "maximum validation NRMSE across canonical E1 noise seeds"
            ),
            "passing_seed_count": passing_seed_count,
            "all_seeds_pass": passing_seed_count == len(seed_values),
            "gate": {
                "metric": "minimum_per_seed_support_f1",
                "criterion": "greater_than_or_equal",
                "threshold": SUPPORT_F1_THRESHOLD,
                "observed": minimum_support_f1,
                "pass": minimum_support_f1 >= SUPPORT_F1_THRESHOLD,
            },
        },
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_bytes(_json_bytes(metrics))
    return metrics


if __name__ == "__main__":
    result_metrics = run_experiment()
    aggregate = result_metrics["aggregate"]
    print(
        "E1 seeds passing:",
        f"{aggregate['passing_seed_count']}/{len(aggregate['seeds'])}",
        "minimum support F1:",
        f"{aggregate['minimum_support_f1']:.6g}",
    )
