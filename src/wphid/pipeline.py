"""Leakage-safe end-to-end weak-form discovery pipelines."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from wphid.dimensions import validate_equation_dimensions
from wphid.eiv import fit_simex_fixed_support
from wphid.sparse.stlsq import fit_stlsq
from wphid.sparse.subset import best_subset_aic
from wphid.stability import (
    WeakFormCondition,
    aggregate_stability,
    bootstrap_grouped_weak_form,
)
from wphid.stability.selection import run_stability_selection
from wphid.types import Candidate, DiscoveryResult, Trajectory
from wphid.weak.time import project_time_weak


_DEFAULT_WINDOW = 81
_DEFAULT_STRIDE = 20
_SPARSITY_PENALTY = 1.0e-3
_NORMALIZED_RMSE_DEFINITION = "RMS(residual)/RMS(target)"


def _effective_weak_settings(trajectory: Trajectory) -> tuple[int, int]:
    window = min(_DEFAULT_WINDOW, trajectory.t.size)
    stride = min(_DEFAULT_STRIDE, max(1, window - 1))
    return window, stride


def _condition_matrix(
    trajectory: Trajectory, candidates: Sequence[Candidate]
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    for candidate in candidates:
        values = np.asarray(candidate.evaluate(trajectory.states), dtype=float)
        if values.ndim != 1 or values.shape != trajectory.t.shape:
            raise ValueError(
                f"candidate {candidate.name!r} must evaluate to one value per time point"
            )
        features.append(values)

    window, stride = _effective_weak_settings(trajectory)
    return project_time_weak(
        trajectory.t,
        np.asarray(trajectory.states["Do"], dtype=float),
        np.column_stack(features),
        window=window,
        stride=stride,
    )


def _normalized_rmse(matrix: np.ndarray, target: np.ndarray, coefficients: np.ndarray) -> float:
    residual_rms = float(np.sqrt(np.mean((matrix @ coefficients - target) ** 2)))
    target_rms = float(np.sqrt(np.mean(target**2)))
    if target_rms == 0.0:
        return 0.0 if residual_rms == 0.0 else float("inf")
    return residual_rms / target_rms


def _select_trajectories(
    trajectories: Sequence[Trajectory],
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
) -> tuple[list[Trajectory], list[Trajectory]]:
    if not train_ids or not validation_ids:
        raise ValueError("train_ids and validation_ids must both be non-empty")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("train_ids must contain unique condition IDs")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("validation_ids must contain unique condition IDs")
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation condition IDs must be disjoint")

    by_id = {trajectory.condition_id: trajectory for trajectory in trajectories}
    if len(by_id) != len(trajectories):
        raise ValueError("trajectory condition IDs must be unique")
    missing = (set(train_ids) | set(validation_ids)).difference(by_id)
    if missing:
        raise ValueError(f"unknown condition IDs: {', '.join(sorted(missing))}")
    return [by_id[name] for name in train_ids], [by_id[name] for name in validation_ids]


def discover_closed(
    trajectories: Iterable[Trajectory],
    candidates: Sequence[Candidate],
    target_dimension: tuple[int, ...],
    train_ids: Iterable[str],
    validation_ids: Iterable[str],
    thresholds: Iterable[float],
) -> DiscoveryResult:
    """Discover a sparse closed-library model without truth-support leakage.

    Sparse models are fit on training conditions, ranked only by validation
    weak-form error plus a fixed support-size penalty, and then refit on the
    training conditions at the selected support.
    """
    observations = tuple(trajectories)
    library = tuple(candidates)
    training_ids = tuple(train_ids)
    held_out_ids = tuple(validation_ids)
    threshold_values = np.asarray(tuple(thresholds), dtype=float)

    if not observations:
        raise ValueError("at least one trajectory is required")
    if not library:
        raise ValueError("at least one candidate is required")
    names = tuple(candidate.name for candidate in library)
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    equation_target_dimension = validate_equation_dimensions(
        library, target_dimension
    )
    if threshold_values.ndim != 1 or threshold_values.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(threshold_values)) or np.any(threshold_values < 0.0):
        raise ValueError("thresholds must contain finite non-negative values")

    training, validation = _select_trajectories(
        observations, training_ids, held_out_ids
    )
    matrices = {
        trajectory.condition_id: _condition_matrix(trajectory, library)
        for trajectory in training + validation
    }
    train_matrix = np.vstack([matrices[name][0] for name in training_ids])
    weak_settings = {
        trajectory.condition_id: _effective_weak_settings(trajectory)
        for trajectory in training + validation
    }
    weak_window_by_condition = {
        name: settings[0] for name, settings in weak_settings.items()
    }
    weak_stride_by_condition = {
        name: settings[1] for name, settings in weak_settings.items()
    }
    unique_windows = set(weak_window_by_condition.values())
    unique_strides = set(weak_stride_by_condition.values())
    weak_window = next(iter(unique_windows)) if len(unique_windows) == 1 else None
    weak_stride = next(iter(unique_strides)) if len(unique_strides) == 1 else None
    train_target = np.concatenate([matrices[name][1] for name in training_ids])
    validation_matrix = np.vstack([matrices[name][0] for name in held_out_ids])
    validation_target = np.concatenate([matrices[name][1] for name in held_out_ids])

    history: list[dict[str, Any]] = []
    selected_key: tuple[float, int, float] | None = None
    selected_support = np.empty(0, dtype=int)
    selected_threshold = float("nan")
    for threshold in threshold_values:
        fit = fit_stlsq(train_matrix, train_target, float(threshold))
        validation_error = _normalized_rmse(
            validation_matrix, validation_target, fit.coefficients
        )
        score = validation_error + _SPARSITY_PENALTY * fit.support.size
        history.append(
            {
                "threshold": float(threshold),
                "validation_error": validation_error,
                "support_size": int(fit.support.size),
                "selection_score": score,
            }
        )
        key = (score, int(fit.support.size), -float(threshold))
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_support = np.asarray(fit.support, dtype=int)
            selected_threshold = float(threshold)

    coefficients = np.zeros(len(library))
    if selected_support.size:
        coefficients[selected_support] = np.linalg.lstsq(
            train_matrix[:, selected_support], train_target, rcond=None
        )[0]
    validation_error = _normalized_rmse(
        validation_matrix, validation_target, coefficients
    )
    per_condition_error = {
        name: _normalized_rmse(matrix, target, coefficients)
        for name, (matrix, target) in matrices.items()
    }
    diagnostics = {
        "selected_threshold": selected_threshold,
        "selection_score": float(selected_key[0]),
        "sparsity_penalty_per_term": _SPARSITY_PENALTY,
        "train_ids": training_ids,
        "validation_ids": held_out_ids,
        "weak_window": weak_window,
        "weak_stride": weak_stride,
        "weak_window_by_condition": weak_window_by_condition,
        "weak_stride_by_condition": weak_stride_by_condition,
        "normalized_rmse_definition": _NORMALIZED_RMSE_DEFINITION,
        "target_dimension": equation_target_dimension,
        "candidate_dimensions": {
            candidate.name: candidate.dimension for candidate in library
        },
        "coefficient_dimensions": {
            candidate.name: candidate.coefficient_dimension for candidate in library
        },
        "weak_rows_by_condition": {
            name: matrix.shape[0] for name, (matrix, _) in matrices.items()
        },
        "normalized_rmse_by_condition": per_condition_error,
        "threshold_history": history,
    }
    return DiscoveryResult(
        names=names,
        coefficients=coefficients,
        support=selected_support,
        validation_error=validation_error,
        diagnostics=diagnostics,
    )


def _profile_training_score(
    trajectories_by_id: Mapping[str, Trajectory],
    candidates: Sequence[Candidate],
    target_dimension: tuple[int, ...],
    training_ids: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    """Select one profile threshold by leave-one-training-condition-out error."""
    validate_equation_dimensions(candidates, target_dimension)
    matrices = {
        condition_id: _condition_matrix(
            trajectories_by_id[condition_id], candidates
        )
        for condition_id in training_ids
    }
    threshold_history: list[dict[str, Any]] = []
    selected_key: tuple[float, float, float] | None = None
    selected_record: dict[str, Any] | None = None
    for threshold in thresholds:
        fold_errors = {}
        fold_support_sizes = {}
        for held_out_id in training_ids:
            fit_ids = tuple(
                condition_id
                for condition_id in training_ids
                if condition_id != held_out_id
            )
            train_matrix = np.vstack(
                [matrices[condition_id][0] for condition_id in fit_ids]
            )
            train_target = np.concatenate(
                [matrices[condition_id][1] for condition_id in fit_ids]
            )
            fit = fit_stlsq(
                train_matrix,
                train_target,
                float(threshold),
            )
            held_out_matrix, held_out_target = matrices[held_out_id]
            fold_errors[held_out_id] = _normalized_rmse(
                held_out_matrix,
                held_out_target,
                fit.coefficients,
            )
            fold_support_sizes[held_out_id] = int(fit.support.size)
        cross_validation_error = float(
            np.median(tuple(fold_errors.values()))
        )
        median_support_size = float(
            np.median(tuple(fold_support_sizes.values()))
        )
        selection_score = (
            cross_validation_error
            + _SPARSITY_PENALTY * median_support_size
        )
        record = {
            "threshold": float(threshold),
            "fold_errors": fold_errors,
            "fold_support_sizes": fold_support_sizes,
            "cross_validation_error": cross_validation_error,
            "median_support_size": median_support_size,
            "selection_score": selection_score,
        }
        threshold_history.append(record)
        key = (
            selection_score,
            median_support_size,
            -float(threshold),
        )
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_record = record
    if selected_record is None:
        raise ValueError("thresholds must be non-empty")
    return {
        "selected_training_threshold": selected_record["threshold"],
        "training_cross_validation_error": selected_record[
            "cross_validation_error"
        ],
        "median_support_size": selected_record["median_support_size"],
        "selection_score": selected_record["selection_score"],
        "threshold_history": threshold_history,
    }


def discover_profiled_closed(
    trajectories: Iterable[Trajectory],
    candidate_profiles: Mapping[float, Sequence[Candidate]],
    parameter_name: str,
    target_dimension: tuple[int, ...],
    train_ids: Iterable[str],
    validation_ids: Iterable[str],
    thresholds: Iterable[float],
) -> DiscoveryResult:
    """Select a nonlinear parameter profile on training conditions only.

    The selected candidate profile is then passed to ``discover_closed``,
    where the ordinary sparse threshold selection may use validation data.
    Neither validation observations nor truth parameters enter profile
    selection.
    """
    if not isinstance(parameter_name, str) or not parameter_name:
        raise ValueError("parameter_name must be a non-empty string")
    observations = tuple(trajectories)
    training_ids = tuple(train_ids)
    held_out_ids = tuple(validation_ids)
    if len(training_ids) < 2:
        raise ValueError(
            "parameter profiling requires at least two training IDs"
        )
    training, _ = _select_trajectories(
        observations, training_ids, held_out_ids
    )
    trajectories_by_id = {
        trajectory.condition_id: trajectory for trajectory in training
    }
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values or not np.all(np.isfinite(threshold_values)):
        raise ValueError("thresholds must contain finite values")
    if any(value < 0.0 for value in threshold_values):
        raise ValueError("thresholds must contain non-negative values")

    normalized_profiles = []
    for raw_parameter, raw_candidates in candidate_profiles.items():
        parameter = float(raw_parameter)
        if not np.isfinite(parameter):
            raise ValueError("profile parameter values must be finite")
        candidates = tuple(raw_candidates)
        if not candidates:
            raise ValueError("every parameter profile needs candidates")
        names = tuple(candidate.name for candidate in candidates)
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique within a profile")
        normalized_profiles.append((parameter, candidates))
    if not normalized_profiles:
        raise ValueError("candidate_profiles must be non-empty")
    normalized_profiles.sort(key=lambda item: item[0])

    profile_history = []
    selected_key: tuple[float, float, float, float] | None = None
    selected_parameter = float("nan")
    selected_candidates: tuple[Candidate, ...] = ()
    for parameter, candidates in normalized_profiles:
        profile = _profile_training_score(
            trajectories_by_id,
            candidates,
            target_dimension,
            training_ids,
            threshold_values,
        )
        profile_record = {
            "parameter": parameter,
            "candidate_identities": tuple(
                candidate.canonical_identity for candidate in candidates
            ),
            **profile,
        }
        profile_history.append(profile_record)
        key = (
            profile["selection_score"],
            profile["training_cross_validation_error"],
            profile["median_support_size"],
            parameter,
        )
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_parameter = parameter
            selected_candidates = candidates

    result = discover_closed(
        trajectories=observations,
        candidates=selected_candidates,
        target_dimension=target_dimension,
        train_ids=training_ids,
        validation_ids=held_out_ids,
        thresholds=threshold_values,
    )
    diagnostics = {
        **dict(result.diagnostics),
        "parameter_name": parameter_name,
        "selected_parameter": selected_parameter,
        "parameter_selection_data": "training_conditions_only",
        "parameter_selection_rule": (
            "leave_one_training_condition_out_profile_score"
        ),
        "parameter_selection_train_ids": training_ids,
        "parameter_profile_history": profile_history,
    }
    return DiscoveryResult(
        names=result.names,
        coefficients=result.coefficients,
        support=result.support,
        validation_error=result.validation_error,
        diagnostics=diagnostics,
    )


def discover_robust_profiled_closed(
    trajectories, candidate_profiles, parameter_name, target_dimension,
    train_ids, validation_ids, thresholds, *, n_resamples, block_length,
    random_seed, simex_levels, simex_replicates,
    observation_noise_reconstructor, inclusion_threshold=0.8,
    sign_consistency_threshold=0.8,
    simex_extrapolation_condition_limit=1.0e8,
    selection_sparsity_penalty=0.01,
) -> DiscoveryResult:
    """Select stable train-only support, then perform frozen-support SIMEX."""
    observations = tuple(trajectories)
    training_ids, held_out_ids = tuple(train_ids), tuple(validation_ids)
    training, validation = _select_trajectories(observations, training_ids, held_out_ids)
    profiles = {float(key): tuple(value) for key, value in candidate_profiles.items()}
    if not profiles:
        raise ValueError("candidate_profiles must be non-empty")
    ordered = tuple(sorted(profiles.items()))
    names = tuple(item.canonical_identity for item in ordered[0][1])
    if any(
        tuple(item.canonical_identity for item in library) != names
        for _, library in ordered
    ):
        raise ValueError(
            "all candidate profiles must share canonical identities and ordering"
        )
    for _, library in ordered:
        validate_equation_dimensions(library, target_dimension)
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values:
        raise ValueError("thresholds must be non-empty")
    matrices_by_profile = {
        parameter: {
            item.condition_id: _condition_matrix(item, library)
            for item in training
        }
        for parameter, library in ordered
    }
    reference_parameter = ordered[0][0]
    weak_conditions = {
        condition_id: WeakFormCondition(condition_id, matrix, target)
        for condition_id, (matrix, target) in
        matrices_by_profile[reference_parameter].items()
    }
    rng = np.random.default_rng(random_seed)
    bootstrap_coefficients = []
    bootstrap_parameters = []
    for _ in range(n_resamples):
        sample = bootstrap_grouped_weak_form(
            weak_conditions,
            block_length=block_length,
            random_seed=int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)),
        )
        best = None
        for parameter, _ in ordered:
            profile_matrices = matrices_by_profile[parameter]
            matrix = np.vstack([
                profile_matrices[draw.source_condition_id][0][draw.row_indices]
                for draw in sample.conditions
            ])
            target = np.concatenate([
                profile_matrices[draw.source_condition_id][1][draw.row_indices]
                for draw in sample.conditions
            ])
            for threshold in threshold_values:
                fit = fit_stlsq(matrix, target, threshold)
                evaluation_matrix = np.vstack([
                    pair[0] for pair in profile_matrices.values()
                ])
                evaluation_target = np.concatenate([
                    pair[1] for pair in profile_matrices.values()
                ])
                evaluation_error = _normalized_rmse(
                    evaluation_matrix, evaluation_target, fit.coefficients
                )
                score = (
                    evaluation_error
                    + selection_sparsity_penalty * fit.support.size
                )
                key = (score, fit.support.size, parameter, -threshold)
                if best is None or key < best[0]:
                    best = (key, fit.coefficients, parameter)
        bootstrap_coefficients.append(best[1])
        bootstrap_parameters.append(best[2])
    stability = aggregate_stability(
        np.asarray(bootstrap_coefficients),
        candidate_names=names,
        inclusion_threshold=inclusion_threshold,
        sign_consistency_threshold=sign_consistency_threshold,
        theta_values=bootstrap_parameters,
    )
    if stability.support.size == 0:
        raise ValueError("stability selection produced an empty support")
    frequencies = dict(stability.parameter_selection_frequencies)
    selected_parameter = max(frequencies, key=lambda value: (frequencies[value], -value))
    selected_candidates = profiles[selected_parameter]
    output_names = tuple(item.name for item in selected_candidates)

    def reconstruction(level, replicate):
        return observation_noise_reconstructor(
            level, replicate, training_ids, selected_candidates
        )

    simex = fit_simex_fixed_support(
        tuple(simex_levels), reconstruction, stability.support,
        n_replicates=simex_replicates,
        extrapolation_condition_limit=simex_extrapolation_condition_limit,
    )
    coefficients = np.zeros(len(names))
    coefficients[stability.support] = simex.extrapolated_coefficients
    pairs = [_condition_matrix(item, selected_candidates) for item in validation]
    validation_error = _normalized_rmse(
        np.vstack([matrix for matrix, _ in pairs]),
        np.concatenate([target for _, target in pairs]), coefficients,
    )
    extrapolation = dict(simex.diagnostics["extrapolation"])
    diagnostics = {
        "parameter_name": parameter_name,
        "selected_parameter": selected_parameter,
        "train_ids": training_ids,
        "validation_ids": held_out_ids,
        "validation_data_role": "frozen_model_evaluation_only",
        "inclusion_probability": dict(zip(names, stability.inclusion_probabilities, strict=True)),
        "sign_consistency": dict(zip(names, stability.sign_consistency, strict=True)),
        "theta_frequency": frequencies,
        "canonical_identities": names,
        "support_stable": bool(stability.support.size),
        "fixed_support_identifiable": bool(
            simex.diagnostics["identifiability"]["identifiable"]
        ),
        "simex_extrapolation_stable": bool(extrapolation["stable"]),
        "stable": bool(
            stability.support.size
            and simex.diagnostics["identifiability"]["identifiable"]
            and extrapolation["stable"]
        ),
        "simex_fallback": not bool(extrapolation["stable"]),
        "identifiability_scope": "fixed_support_simex_only",
        "simex_diagnostics": dict(simex.diagnostics),
    }
    return DiscoveryResult(
        output_names, coefficients, stability.support, validation_error, diagnostics
    )


def _subset_rss(
    matrix: np.ndarray, target: np.ndarray, support: np.ndarray
) -> float:
    """Ordinary-least-squares residual sum of squares on a fixed support."""
    if support.size == 0:
        return float(np.dot(target, target))
    coefficients = np.linalg.lstsq(matrix[:, support], target, rcond=None)[0]
    residual = matrix[:, support] @ coefficients - target
    return float(np.dot(residual, residual))


@dataclass(frozen=True)
class ThetaProfileResult:
    theta_star: float
    support_star: np.ndarray
    weights: tuple[float, ...]
    aic_history: list[dict[str, Any]]
    refinement_history: list[dict[str, Any]]
    theta_refinement_at_boundary: bool
    theta_bracket: tuple[float, float]


def select_theta_profile(
    training: Sequence[Trajectory],
    library_builder: Callable[[float], Sequence[Candidate]],
    theta_grid: Iterable[float],
    *,
    normalize_per_condition: bool = True,
    theta_bracket: tuple[float, float] | None = None,
) -> ThetaProfileResult:
    """Select theta by grid AIC followed by continuous profile refinement."""
    grid = tuple(sorted(float(value) for value in theta_grid))
    if len(grid) < 2:
        raise ValueError("theta_grid must contain at least two values")
    if not np.all(np.isfinite(grid)) or any(value <= 0.0 for value in grid):
        raise ValueError("theta_grid must contain finite positive values")
    reference_library = tuple(library_builder(grid[0]))
    weights = []
    for trajectory in training:
        _, target = _condition_matrix(trajectory, reference_library)
        weights.append(1.0 / max(float(np.sqrt(np.mean(target**2))), 1e-300))

    def stacked(library: Sequence[Candidate], apply_weights: bool):
        matrices, targets = [], []
        scale = weights if apply_weights else [1.0] * len(training)
        for trajectory, weight in zip(training, scale):
            matrix, target = _condition_matrix(trajectory, library)
            matrices.append(weight * matrix)
            targets.append(weight * target)
        return np.vstack(matrices), np.concatenate(targets)

    aic_history = []
    best_grid = None
    for theta in grid:
        A, b = stacked(tuple(library_builder(theta)), normalize_per_condition)
        support, aic = best_subset_aic(A, b)
        aic_history.append({"theta": theta, "support": support.tolist(), "aic": aic})
        key = (aic, theta)
        if best_grid is None or key < best_grid[0]:
            best_grid = (key, theta, support)
    if best_grid is None or best_grid[2].size == 0:
        raise ValueError("theta grid produced no nonempty candidate support")
    theta_star, support_star = float(best_grid[1]), best_grid[2].copy()
    span = float(grid[-1] - grid[0])
    if theta_bracket is None:
        lower, upper = max(grid[0] - 0.5 * span, 1.0), grid[-1] + 0.5 * span
    else:
        lower, upper = map(float, theta_bracket)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("theta_bracket must satisfy 0 < lower < upper")
    if not lower < theta_star < upper:
        pad = 0.1 * span
        lower, upper = min(lower, theta_star - pad), max(upper, theta_star + pad)
    refinement_history = []
    median_spacing = float(np.median(np.diff(grid)))
    for _ in range(3):
        def objective(candidate_theta: float) -> float:
            A, b = stacked(tuple(library_builder(candidate_theta)), normalize_per_condition)
            return _subset_rss(A, b, support_star)
        fit = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 25.0})
        candidate_theta = float(fit.x)
        A, b = stacked(tuple(library_builder(candidate_theta)), normalize_per_condition)
        candidate_support, _ = best_subset_aic(A, b)
        refinement_history.append({"theta": candidate_theta, "support": candidate_support.tolist(), "rss": float(fit.fun)})
        support_changed = not np.array_equal(candidate_support, support_star)
        theta_drift = abs(candidate_theta - theta_star)
        theta_star, support_star = candidate_theta, candidate_support
        if not support_changed and theta_drift < 0.25 * median_spacing:
            break
    if support_star.size == 0:
        raise ValueError("alternating refinement produced an empty support")
    at_boundary = theta_star < lower + 0.05 * span or theta_star > upper - 0.05 * span
    return ThetaProfileResult(theta_star, support_star, tuple(weights), aic_history,
                              refinement_history, bool(at_boundary), (lower, upper))


def discover_theta_profiled_closed(
    trajectories: Iterable[Trajectory],
    theta_grid: Iterable[float],
    library_builder: Callable[[float], Sequence[Candidate]],
    parameter_name: str,
    target_dimension: tuple[int, ...],
    train_ids: Iterable[str],
    validation_ids: Iterable[str],
    *,
    n_resamples: int,
    block_length: int,
    random_seed: int,
    simex_levels: Iterable[float],
    simex_replicates: int,
    observation_noise_reconstructor: Callable[..., tuple[np.ndarray, np.ndarray]],
    inclusion_threshold: float = 0.8,
    sign_consistency_threshold: float = 0.8,
    theta_bracket: tuple[float, float] | None = None,
    normalize_per_condition: bool = True,
    simex_extrapolation_condition_limit: float = 1.0e8,
) -> DiscoveryResult:
    """Select a nonlinear parameter ``theta`` by continuous profile likelihood
    and its support by exact best-subset AIC, then run frozen-support SIMEX.

    Unlike the grid-modal selection in ``discover_robust_profiled_closed``, the
    parameter is refined on a continuous bracket, so a near-collinear Arrhenius
    grid cannot reduce selection to a coin flip. Training-only data drives every
    selection step; validation enters only for final evaluation.
    """
    observations = tuple(trajectories)
    training_ids, held_out_ids = tuple(train_ids), tuple(validation_ids)
    training, validation = _select_trajectories(
        observations, training_ids, held_out_ids
    )

    if not isinstance(parameter_name, str) or not parameter_name:
        raise ValueError("parameter_name must be a non-empty string")
    grid = tuple(sorted(float(value) for value in theta_grid))
    if len(grid) < 2:
        raise ValueError("theta_grid must contain at least two values")
    if not np.all(np.isfinite(grid)) or any(value <= 0.0 for value in grid):
        raise ValueError("theta_grid must contain finite positive values")

    reference_library = tuple(library_builder(grid[0]))
    names = tuple(item.canonical_identity for item in reference_library)
    if not names or any(name is None for name in names):
        raise ValueError(
            "library candidates must declare canonical identities for cross-θ checks"
        )
    ordered_libraries: dict[float, tuple[Candidate, ...]] = {}
    for theta in grid:
        library = tuple(library_builder(theta))
        if tuple(item.canonical_identity for item in library) != names:
            raise ValueError(
                "library_builder must preserve canonical identities and "
                "ordering across the theta grid"
            )
        validate_equation_dimensions(library, target_dimension)
        ordered_libraries[theta] = library

    profile = select_theta_profile(
        training,
        library_builder,
        grid,
        normalize_per_condition=normalize_per_condition,
        theta_bracket=theta_bracket,
    )
    theta_star = profile.theta_star
    support_star = profile.support_star
    weights = list(profile.weights)
    aic_history = profile.aic_history
    refinement_history = profile.refinement_history
    theta_refinement_at_boundary = profile.theta_refinement_at_boundary
    lower, upper = profile.theta_bracket

    def stacked(
        library: Sequence[Candidate], apply_weights: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        matrices, targets = [], []
        scale = weights if apply_weights else [1.0] * len(training)
        for trajectory, weight in zip(training, scale):
            matrix, target = _condition_matrix(trajectory, library)
            matrices.append(weight * matrix)
            targets.append(weight * target)
        return np.vstack(matrices), np.concatenate(targets)

    selected_library = tuple(library_builder(theta_star))
    output_names = tuple(item.name for item in selected_library)

    # Frozen physical refit (level-0 estimate; SIMEX refines these).
    A_phys, b_phys = stacked(selected_library, apply_weights=False)
    coefficients = np.zeros(len(output_names))
    coefficients[support_star] = np.linalg.lstsq(
        A_phys[:, support_star], b_phys, rcond=None
    )[0]

    # Bootstrap stability diagnostics at the frozen theta, on the same
    # normalized conditions the selector uses (best-subset AIC).
    weak_conditions: dict[str, WeakFormCondition] = {}
    for trajectory, weight in zip(training, weights):
        matrix, target = _condition_matrix(trajectory, selected_library)
        weak_conditions[trajectory.condition_id] = WeakFormCondition(
            trajectory.condition_id, weight * matrix, weight * target
        )

    def aic_selector(
        matrix: np.ndarray,
        target: np.ndarray,
        seed: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        support, _ = best_subset_aic(matrix, target)
        coefficients_out = np.zeros(matrix.shape[1])
        if support.size:
            coefficients_out[support] = np.linalg.lstsq(
                matrix[:, support], target, rcond=None
            )[0]
        return coefficients_out

    stability = run_stability_selection(
        weak_conditions,
        candidate_names=output_names,
        selector=aic_selector,
        n_resamples=n_resamples,
        block_length=block_length,
        random_seed=int(random_seed),
        inclusion_threshold=inclusion_threshold,
        sign_consistency_threshold=sign_consistency_threshold,
    )

    def reconstruction(level: float, replicate: int):
        return observation_noise_reconstructor(
            level, replicate, training_ids, selected_library
        )

    simex = fit_simex_fixed_support(
        tuple(simex_levels),
        reconstruction,
        support_star,
        n_replicates=simex_replicates,
        extrapolation_condition_limit=simex_extrapolation_condition_limit,
    )
    coefficients[support_star] = simex.extrapolated_coefficients

    pairs = [_condition_matrix(item, selected_library) for item in validation]
    validation_error = _normalized_rmse(
        np.vstack([matrix for matrix, _ in pairs]),
        np.concatenate([target for _, target in pairs]),
        coefficients,
    )

    extrapolation = dict(simex.diagnostics["extrapolation"])
    diagnostics = {
        "parameter_name": parameter_name,
        "selected_parameter": theta_star,
        "theta_selection_data": "training_conditions_only",
        "theta_selection_rule": (
            "best_subset_aic_grid_profile_likelihood_alternating"
        ),
        "theta_grid": grid,
        "theta_bracket": (lower, upper),
        "normalize_per_condition": bool(normalize_per_condition),
        "theta_refinement_at_boundary": theta_refinement_at_boundary,
        "aic_history": aic_history,
        "refinement_history": refinement_history,
        "train_ids": training_ids,
        "validation_ids": held_out_ids,
        "validation_data_role": "frozen_model_evaluation_only",
        "inclusion_probability": dict(
            zip(output_names, stability.inclusion_probabilities, strict=True)
        ),
        "sign_consistency": dict(
            zip(output_names, stability.sign_consistency, strict=True)
        ),
        "theta_frequency": {f"{theta_star:.6g}": 1.0},
        "canonical_identities": names,
        "support_stable": bool(stability.support.size),
        "fixed_support_identifiable": bool(
            simex.diagnostics["identifiability"]["identifiable"]
        ),
        "simex_extrapolation_stable": bool(extrapolation["stable"]),
        "stable": bool(
            stability.support.size
            and simex.diagnostics["identifiability"]["identifiable"]
            and extrapolation["stable"]
        ),
        "simex_fallback": not bool(extrapolation["stable"]),
        "coefficient_estimation": (
            "physical_ols_then_simex_at_frozen_support"
        ),
        "identifiability_scope": "fixed_support_simex_only",
        "simex_diagnostics": dict(simex.diagnostics),
    }
    return DiscoveryResult(
        output_names,
        coefficients,
        support_star,
        validation_error,
        diagnostics,
    )
