"""Run the reproducible E3 finite-pool missing-interaction gate."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.e1_closed_library import (
    THETA_GRID,
    THRESHOLDS,
    closed_candidate_library,
)
from wphid.dimensions import DAMAGE_RATE, subtract_dimensions
from wphid.discovery.residual import discover_with_expansion
from wphid.grammar.atoms import PRESSURE, aging_atoms
from wphid.grammar.generate import generate_candidates
from wphid.pipeline import (
    _condition_matrix,
    _normalized_rmse,
    discover_closed,
    discover_profiled_closed,
)
from wphid.truth.aging_ode import simulate_condition
from wphid.types import Candidate, DiscoveryResult, Trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 23, 47, 89, 131)
NOISE_FRACTION = 1.0e-4
TRAIN_IDS = ("fatigue_warm", "fatigue_oxygen")
VALIDATION_IDS = ("fatigue_validation",)
TEST_IDS = ("fatigue_test",)
EXPECTED_MISSING_INTERACTION = "arr(T;theta=1800)*pO2*Df"
DEFAULT_OUTPUT_DIR = Path("artifacts/e3_missing_interaction")
DEFAULT_E1_METRICS_PATH = PROJECT_ROOT / "artifacts/e1_closed_library/metrics.json"
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
_SCOPE = {
    "admission": "finite-pool validation admission",
    "claim_boundary": (
        "controlled finite-pool evidence; not unrestricted grammar discovery"
    ),
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
        "fatigue_test": {
            "T": 401.0 + 13.0 * np.cos(1.5 * phase + 0.6),
            "pO2": 0.36 + 0.08 * np.sin(2.5 * phase + 0.1),
            "sigma": 50.0 + 24.0 * np.sin(4.5 * phase + 1.0),
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


def _is_declared_missing_form(candidate: Candidate) -> bool:
    non_arrhenius_parents = {
        parent for parent in candidate.parents if not parent.startswith("arr(T;")
    }
    return (
        len(candidate.parents) == 3
        and len(non_arrhenius_parents) == 2
        and non_arrhenius_parents == {"pO2", "Df"}
    )


def _candidate_libraries(
    arrhenius_temperature: float,
) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """Build one declared theta profile without consulting evaluation truth."""
    generated = generate_candidates(
        aging_atoms(arrhenius_temperature=arrhenius_temperature),
        PRESSURE,
        coefficient_dimension=subtract_dimensions(DAMAGE_RATE, PRESSURE),
    )
    closed = closed_candidate_library(arrhenius_temperature)
    saturation_distractor = next(
        candidate
        for candidate in generated
        if candidate.name.startswith("arr(T;")
        and candidate.name.endswith("*pO2*sat(Df)")
    )
    if saturation_distractor.name not in {
        candidate.name for candidate in closed
    }:
        closed = [*closed, saturation_distractor]
    fixed = [
        candidate
        for candidate in closed
        if not _is_declared_missing_form(candidate)
    ]
    fixed_names = {candidate.name for candidate in fixed}
    expansion = [
        candidate
        for candidate in generated
        if candidate.group == "G3" and candidate.name not in fixed_names
    ]
    return closed, fixed, expansion


def _theta_candidate_profiles() -> dict[float, list[Candidate]]:
    """Profile theta on training over the declared finite candidate pool."""
    profiles = {}
    for raw_theta in THETA_GRID:
        theta = float(raw_theta)
        _, fixed, expansion = _candidate_libraries(theta)
        profiles[theta] = [*fixed, *expansion]
    return profiles


def _select_theta(
    trajectories: Sequence[Trajectory],
) -> tuple[float, dict[str, Any]]:
    profiled = discover_profiled_closed(
        trajectories=trajectories,
        candidate_profiles=_theta_candidate_profiles(),
        parameter_name="arrhenius_theta",
        target_dimension=DAMAGE_RATE,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    selected_theta = float(profiled.diagnostics["selected_parameter"])
    profile_history = [
        {
            "theta": float(profile["parameter"]),
            "selected_training_threshold": float(
                profile["selected_training_threshold"]
            ),
            "training_cross_validation_error": float(
                profile["training_cross_validation_error"]
            ),
            "median_support_size": float(profile["median_support_size"]),
            "selection_score": float(profile["selection_score"]),
        }
        for profile in profiled.diagnostics["parameter_profile_history"]
    ]
    evidence = {
        "parameter_name": "arrhenius_theta",
        "selected_theta": selected_theta,
        "selection_data": profiled.diagnostics["parameter_selection_data"],
        "selection_rule": profiled.diagnostics["parameter_selection_rule"],
        "train_ids": list(
            profiled.diagnostics["parameter_selection_train_ids"]
        ),
        "profile_history": profile_history,
    }
    return selected_theta, evidence


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


def _test_error(
    result: DiscoveryResult,
    candidate_pool: Sequence[Candidate],
    trajectories: Sequence[Trajectory],
) -> float:
    candidates_by_name = {candidate.name: candidate for candidate in candidate_pool}
    selected_candidates = [candidates_by_name[name] for name in result.names]
    trajectories_by_id = {
        trajectory.condition_id: trajectory for trajectory in trajectories
    }
    matrices = [
        _condition_matrix(
            trajectories_by_id[condition_id], selected_candidates
        )
        for condition_id in TEST_IDS
    ]
    return _normalized_rmse(
        np.vstack([matrix for matrix, _ in matrices]),
        np.concatenate([target for _, target in matrices]),
        result.coefficients,
    )


def _result_summary(
    result: DiscoveryResult,
    test_normalized_rmse: float,
) -> dict[str, Any]:
    return {
        "candidate_names": list(result.names),
        "support": _support_names(result),
        "coefficients": {
            name: float(coefficient)
            for name, coefficient in zip(
                result.names, result.coefficients, strict=True
            )
        },
        "selected_threshold": float(result.diagnostics["selected_threshold"]),
        "validation_normalized_rmse": float(result.validation_error),
        "test_normalized_rmse": float(test_normalized_rmse),
    }


def _e1_source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_e1_reference(
    metrics_path: str | Path = DEFAULT_E1_METRICS_PATH,
) -> dict[str, Any]:
    """Load the E1 gate and NRMSE from one explicit generated JSON artifact."""
    path = Path(metrics_path)
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        aggregate = metrics["aggregate"]
        gate = aggregate["gate"]
        gate_pass = gate["pass"]
        reference_nrmse = float(
            aggregate["reference_validation_normalized_rmse"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid generated E1 metrics artifact: {path}"
        ) from error
    if metrics.get("experiment") != "e1_closed_library":
        raise ValueError("E1 metrics artifact has the wrong experiment name")
    if not isinstance(gate_pass, bool):
        raise ValueError("E1 gate pass must be boolean")
    if not np.isfinite(reference_nrmse) or reference_nrmse < 0.0:
        raise ValueError("E1 reference validation NRMSE must be finite and non-negative")
    return {
        "source": _e1_source_label(path),
        "gate": _jsonable(gate),
        "gate_pass": gate_pass,
        "all_seeds_pass": bool(aggregate["all_seeds_pass"]),
        "reference_validation_normalized_rmse": reference_nrmse,
        "reference_validation_nrmse_definition": aggregate.get(
            "reference_validation_nrmse_definition"
        ),
    }


def run_seed(
    seed: int,
    *,
    e1_reference: Mapping[str, Any] | None = None,
    trajectories: Sequence[Trajectory] | None = None,
) -> dict[str, Any]:
    """Run one E3 replicate with truth used only by the final evaluator."""
    observations = (
        list(trajectories) if trajectories is not None else _observations(seed)
    )
    reference = dict(
        load_e1_reference() if e1_reference is None else e1_reference
    )
    e1_nrmse = float(reference["reference_validation_normalized_rmse"])

    selected_theta, parameter_selection = _select_theta(observations)
    closed_candidates, fixed_candidates, expansion_candidates = (
        _candidate_libraries(selected_theta)
    )
    same_seed_closed_result = discover_closed(
        trajectories=observations,
        candidates=closed_candidates,
        target_dimension=DAMAGE_RATE,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    fixed_result = discover_closed(
        trajectories=observations,
        candidates=fixed_candidates,
        target_dimension=DAMAGE_RATE,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        thresholds=THRESHOLDS,
    )
    expanded_result = discover_with_expansion(
        trajectories=observations,
        base_candidates=fixed_candidates,
        expansion_candidates=expansion_candidates,
        train_ids=TRAIN_IDS,
        validation_ids=VALIDATION_IDS,
        test_ids=TEST_IDS,
        target_dimension=DAMAGE_RATE,
        thresholds=THRESHOLDS,
        max_rounds=2,
    )

    same_seed_closed_test_error = _test_error(
        same_seed_closed_result,
        closed_candidates,
        observations,
    )
    fixed_test_error = _test_error(
        fixed_result,
        fixed_candidates,
        observations,
    )
    expanded_test_error = float(
        expanded_result.diagnostics["test_normalized_rmse"]
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

    same_seed_closed_summary = _result_summary(
        same_seed_closed_result,
        same_seed_closed_test_error,
    )
    same_seed_closed_summary["role"] = (
        "post-selection closed-library diagnostic on the same noisy seed"
    )
    fixed_summary = _result_summary(fixed_result, fixed_test_error)
    expanded_summary = _result_summary(expanded_result, expanded_test_error)
    expanded_summary["test_nrmse_over_e1_reference"] = (
        expanded_test_error / e1_nrmse
    )
    gates = {
        "fixed_test_error_at_least_50_percent_above_closed_diagnostic": (
            fixed_test_error >= 1.5 * same_seed_closed_test_error
        ),
        "structured_cross_condition_residual_evidence": structured_evidence,
        "missing_interaction_admitted": (
            EXPECTED_MISSING_INTERACTION in admitted
        ),
        "expanded_test_error_within_20_percent_of_e1_reference": (
            expanded_test_error <= 1.2 * e1_nrmse
        ),
        "at_most_one_false_interaction": len(false_interactions) <= 1,
    }
    return {
        "seed": int(seed),
        "configuration": {
            "observation_noise": dict(_OBSERVATION_NOISE),
            "json_newline": _JSON_NEWLINE,
            "condition_roles": {
                "residual_screening_and_training": list(TRAIN_IDS),
                "finite_pool_validation_admission": list(VALIDATION_IDS),
                "independent_test_evaluation": list(TEST_IDS),
            },

            "target_dimension": list(DAMAGE_RATE),
            "base_candidate_names": [
                candidate.name for candidate in fixed_candidates
            ],
            "expansion_candidate_names": [
                candidate.name for candidate in expansion_candidates
            ],
        },
        "parameter_selection": parameter_selection,
        "evaluation_target": {
            "expected_missing_interaction": EXPECTED_MISSING_INTERACTION,
            "use": "post-discovery evaluation only; not supplied to discovery",
        },
        "same_seed_closed_library_diagnostic": same_seed_closed_summary,
        "fixed_library": fixed_summary,
        "expanded_library": expanded_summary,
        "expansion": {
            "admission_name": "finite-pool validation admission",
            "finite_pool": [
                candidate.name for candidate in expansion_candidates
            ],
            "claim_boundary": _SCOPE["claim_boundary"],
            "admitted_candidates": admitted,
            "false_interactions": false_interactions,
            "rounds": rounds,
        },
        "independent_test_evaluation": {
            "stage_name": "independent test evaluation",
            "condition_ids": list(TEST_IDS),
            "fixed_library_normalized_rmse": fixed_test_error,
            "closed_diagnostic_normalized_rmse": same_seed_closed_test_error,
            "expanded_library_normalized_rmse": expanded_test_error,
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
    e1_metrics_path: str | Path = DEFAULT_E1_METRICS_PATH,
) -> dict[str, Any]:
    """Run all E3 replicates and persist one compact JSON metrics artifact."""
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must be non-empty")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    e1_reference = load_e1_reference(e1_metrics_path)
    per_seed = [
        run_seed(seed, e1_reference=e1_reference) for seed in seed_values
    ]
    passing_seed_count = sum(
        all(result["gates"].values()) for result in per_seed
    )
    e3_gate_pass = passing_seed_count == len(seed_values)
    e1_gate_pass = bool(e1_reference["gate_pass"])
    metrics = {
        "experiment": "e3_missing_interaction",
        "configuration": {
            "theta_grid": list(THETA_GRID),
            "thresholds": [float(value) for value in THRESHOLDS],
            "condition_roles": {
                "residual_screening_and_training": list(TRAIN_IDS),
                "finite_pool_validation_admission": list(VALIDATION_IDS),
                "independent_test_evaluation": list(TEST_IDS),
            },
            "target_dimension": list(DAMAGE_RATE),
        },
        "scope": dict(_SCOPE),
        "e1_reference": e1_reference,
        "per_seed": per_seed,
        "aggregate": {
            "seeds": list(seed_values),
            "passing_seed_count": passing_seed_count,
            "e3_gate_pass": e3_gate_pass,
            "all_seeds_pass": e3_gate_pass,
            "e1_gate_pass": e1_gate_pass,
            "overall_phase_decision": (
                "GO" if e1_gate_pass and e3_gate_pass else "NO-GO"
            ),
            "overall_phase_decision_rule": (
                "E1 and E3 must both pass; E3 cannot override E1"
            ),
            "e1_reference_validation_normalized_rmse": e1_reference[
                "reference_validation_normalized_rmse"
            ],
            "maximum_expanded_test_nrmse_over_e1_reference": max(
                result["expanded_library"][
                    "test_nrmse_over_e1_reference"
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
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_bytes(_json_bytes(metrics))
    return metrics
