"""Leakage-safe end-to-end weak-form discovery pipelines."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from wphid.dimensions import validate_equation_dimensions
from wphid.sparse.stlsq import fit_stlsq
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
