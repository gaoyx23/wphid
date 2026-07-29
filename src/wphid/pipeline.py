"""Leakage-safe end-to-end weak-form discovery pipelines."""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from wphid.sparse.stlsq import fit_stlsq
from wphid.types import Candidate, DiscoveryResult, Trajectory
from wphid.weak.time import project_time_weak


_DEFAULT_WINDOW = 81
_DEFAULT_STRIDE = 20
_SPARSITY_PENALTY = 1.0e-3


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

    window = min(_DEFAULT_WINDOW, trajectory.t.size)
    stride = min(_DEFAULT_STRIDE, max(1, window - 1))
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
        "weak_window": _DEFAULT_WINDOW,
        "weak_stride": _DEFAULT_STRIDE,
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
