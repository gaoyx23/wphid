"""Scale-invariant sequentially thresholded least squares."""

from typing import Any

import numpy as np

from wphid.types import DiscoveryResult


def _as_regression_inputs(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(A, dtype=float)
    target = np.asarray(b, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("A must be two-dimensional")
    if target.ndim != 1:
        raise ValueError("b must be one-dimensional")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("A and b must have the same number of rows")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("A and b must contain only finite values")
    return matrix, target


def _least_squares(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(matrix, target, rcond=None)[0]


def fit_stlsq(
    A: np.ndarray, b: np.ndarray, threshold: float, max_iter: int = 20
) -> DiscoveryResult:
    """Fit a sparse linear model with non-centred scale-invariant selection.

    Columns with zero variance cannot be scaled and are excluded from sparse
    selection.  The selected columns are finally refit against the unscaled
    matrix, so returned coefficients remain in their physical units.
    """
    matrix, target = _as_regression_inputs(A, b)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")

    column_scales = np.std(matrix, axis=0)
    active_columns = np.flatnonzero(column_scales > 0)
    excluded_columns = np.flatnonzero(column_scales == 0)
    target_scale = float(np.std(target))
    if target_scale == 0:
        raise ValueError("b must have non-zero variance for scaling")

    full_coefficients = np.zeros(matrix.shape[1])
    support = np.empty(0, dtype=int)
    scaled_condition = float("nan")
    raw_condition = float("nan")
    iterations = 0
    if active_columns.size:
        scaled_matrix = matrix[:, active_columns] / column_scales[active_columns]
        scaled_target = target / target_scale
        scaled_coefficients = _least_squares(scaled_matrix, scaled_target)
        selected = np.abs(scaled_coefficients) >= threshold
        scaled_condition = float(np.linalg.cond(scaled_matrix))

        for iterations in range(1, max_iter + 1):
            selected_indices = np.flatnonzero(selected)
            if selected_indices.size == 0:
                break
            refit = _least_squares(
                scaled_matrix[:, selected_indices], scaled_target
            )
            updated = np.zeros_like(scaled_coefficients)
            updated[selected_indices] = refit
            next_selected = np.abs(updated) >= threshold
            scaled_coefficients = updated
            if np.array_equal(next_selected, selected):
                selected = next_selected
                break
            selected = next_selected

        support = active_columns[np.flatnonzero(selected)]
        if support.size:
            raw_selected = matrix[:, support]
            full_coefficients[support] = _least_squares(raw_selected, target)
            raw_condition = float(np.linalg.cond(raw_selected))

    residual = matrix @ full_coefficients - target
    validation_error = float(np.linalg.norm(residual) / np.linalg.norm(target))
    diagnostics: dict[str, Any] = {
        "condition_number": raw_condition,
        "scaled_condition_number": scaled_condition,
        "excluded_zero_variance_columns": excluded_columns,
        "column_standard_deviations": column_scales,
        "target_standard_deviation": target_scale,
        "iterations": iterations,
    }
    return DiscoveryResult(
        names=tuple(f"term_{index}" for index in range(matrix.shape[1])),
        coefficients=full_coefficients,
        support=support,
        validation_error=validation_error,
        diagnostics=diagnostics,
    )
