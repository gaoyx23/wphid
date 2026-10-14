"""Pure, fixed-support SIMEX refitting utilities.

The routines here intentionally know nothing about library construction or
support selection.  A caller supplies a frozen set of column indices and, for
each SIMEX noise level and replicate, the already rebuilt weak-form design
matrix and target.  This keeps validation/test data and truth labels outside
the refit boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike


RegressionSample: TypeAlias = tuple[ArrayLike, ArrayLike]
"""One rebuilt full design matrix and its corresponding target."""

ReconstructionCallback: TypeAlias = Callable[[float, int], RegressionSample]
"""Build ``(design, target)`` for a SIMEX level and zero-based replicate."""


def _readonly_float_array(value: ArrayLike) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze(value) for key, value in values.items()})


@dataclass(frozen=True)
class SIMEXRefitResult:
    """Coefficient paths and guarded extrapolation for one frozen support.

    ``coefficient_path`` contains the mean fixed-support OLS coefficients at
    each supplied nonnegative SIMEX level.  Its columns correspond exactly to
    ``support``.  ``extrapolated_coefficients`` is the polynomial estimate at
    level ``-1`` when that extrapolation is stable; otherwise it is the
    uncorrected level-zero estimate.
    """

    support: np.ndarray
    levels: np.ndarray
    coefficient_path: np.ndarray
    coefficient_standard_errors: np.ndarray
    replicate_coefficients: tuple[np.ndarray, ...]
    extrapolated_coefficients: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        support = np.array(self.support, copy=True)
        levels = _readonly_float_array(self.levels)
        path = _readonly_float_array(self.coefficient_path)
        errors = _readonly_float_array(self.coefficient_standard_errors)
        extrapolated = _readonly_float_array(self.extrapolated_coefficients)
        replicates = tuple(_readonly_float_array(value) for value in self.replicate_coefficients)

        if support.ndim != 1 or not np.issubdtype(support.dtype, np.integer):
            raise ValueError("support must be a one-dimensional integer array")
        support = support.astype(np.intp, copy=False)
        if support.size == 0 or np.any(support < 0):
            raise ValueError("support must contain at least one non-negative index")
        if support.size > 1 and not np.all(np.diff(support) > 0):
            raise ValueError("support indices must be strictly increasing")
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("levels must be a non-empty one-dimensional array")
        if path.shape != (levels.size, support.size):
            raise ValueError("coefficient_path shape must be (n_levels, n_support)")
        if errors.shape != path.shape or extrapolated.shape != (support.size,):
            raise ValueError("coefficient result shapes do not match the frozen support")
        if len(replicates) != levels.size or any(
            item.ndim != 2 or item.shape[1] != support.size for item in replicates
        ):
            raise ValueError("replicate coefficients must align with levels and support")
        if not all(np.isfinite(item).all() for item in (levels, path, errors, extrapolated)):
            raise ValueError("SIMEX result arrays must be finite")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

        support.setflags(write=False)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "coefficient_path", path)
        object.__setattr__(self, "coefficient_standard_errors", errors)
        object.__setattr__(self, "replicate_coefficients", replicates)
        object.__setattr__(self, "extrapolated_coefficients", extrapolated)
        object.__setattr__(self, "diagnostics", _immutable_mapping(self.diagnostics))


def _validate_levels(levels: ArrayLike) -> np.ndarray:
    values = np.asarray(levels, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("levels must contain at least two one-dimensional entries")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("levels must be finite and non-negative")
    if not np.any(values == 0.0):
        raise ValueError("levels must include 0.0")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("levels must be strictly increasing")
    return values


def _validate_support(support: ArrayLike) -> np.ndarray:
    raw = np.asarray(support)
    if raw.ndim != 1 or np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("support must be a one-dimensional integer array")
    indices = raw.astype(np.intp, copy=False)
    if indices.size == 0 or np.any(indices < 0):
        raise ValueError("support must contain at least one non-negative index")
    if indices.size > 1 and not np.all(np.diff(indices) > 0):
        raise ValueError("support indices must be strictly increasing")
    return indices


def _as_regression_sample(
    sample: RegressionSample, support: np.ndarray, expected_columns: int | None
) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        design, target = sample
    except (TypeError, ValueError) as error:
        raise ValueError("each reconstruction must be a (design, target) pair") from error
    matrix = np.asarray(design, dtype=float)
    response = np.asarray(target, dtype=float)
    if matrix.ndim != 2 or response.ndim != 1:
        raise ValueError("design must be two-dimensional and target one-dimensional")
    if matrix.shape[0] != response.size:
        raise ValueError("design and target must have the same number of rows")
    if matrix.shape[0] == 0 or support[-1] >= matrix.shape[1]:
        raise ValueError("frozen support must be valid for every reconstructed design")
    if expected_columns is not None and matrix.shape[1] != expected_columns:
        raise ValueError("every reconstructed design must use the same full column layout")
    if not np.isfinite(matrix).all() or not np.isfinite(response).all():
        raise ValueError("reconstructed designs and targets must be finite")
    return matrix, response, matrix.shape[1]


def _materialize_reconstructions(
    levels: np.ndarray,
    reconstructions: Sequence[Sequence[RegressionSample]] | ReconstructionCallback,
    support: np.ndarray,
    n_replicates: int | None,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]:
    if callable(reconstructions):
        if not isinstance(n_replicates, int) or isinstance(n_replicates, bool) or n_replicates < 1:
            raise ValueError("callback reconstructions require n_replicates >= 1")
        source: Sequence[Sequence[RegressionSample]] = tuple(
            tuple(reconstructions(float(level), replicate) for replicate in range(n_replicates))
            for level in levels
        )
    else:
        if n_replicates is not None:
            raise ValueError("n_replicates is only used with a reconstruction callback")
        source = reconstructions

    if len(source) != levels.size:
        raise ValueError("reconstructions must provide one replicate sequence per level")
    expected_columns: int | None = None
    materialized: list[tuple[tuple[np.ndarray, np.ndarray], ...]] = []
    for level_samples in source:
        if len(level_samples) == 0:
            raise ValueError("each SIMEX level must contain at least one replicate")
        converted: list[tuple[np.ndarray, np.ndarray]] = []
        for sample in level_samples:
            matrix, target, expected_columns = _as_regression_sample(
                sample, support, expected_columns
            )
            converted.append((matrix, target))
        materialized.append(tuple(converted))
    return tuple(materialized)


def _fit_path(
    reconstructions: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...], support: np.ndarray
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    replicate_coefficients: list[np.ndarray] = []
    path: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    for level_samples in reconstructions:
        coefficients: list[np.ndarray] = []
        level_ranks: list[int] = []
        level_conditions: list[float] = []
        for matrix, target in level_samples:
            selected = matrix[:, support]
            coefficient, _, rank, _ = np.linalg.lstsq(selected, target, rcond=None)
            coefficients.append(coefficient)
            level_ranks.append(rank)
            level_conditions.append(float(np.linalg.cond(selected)))
        values = np.vstack(coefficients)
        replicate_coefficients.append(values)
        path.append(np.mean(values, axis=0))
        errors.append(np.std(values, axis=0, ddof=0) / np.sqrt(values.shape[0]))
        ranks.append(np.asarray(level_ranks, dtype=int))
        conditions.append(np.asarray(level_conditions, dtype=float))
    return (
        tuple(replicate_coefficients),
        np.vstack(path),
        np.vstack(errors),
        tuple(ranks),
        tuple(conditions),
    )


def _extrapolate_or_fallback(
    levels: np.ndarray,
    path: np.ndarray,
    degree: int,
    condition_limit: float,
    fixed_support_identifiable: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    zero_index = int(np.flatnonzero(levels == 0.0)[0])
    fallback = path[zero_index].copy()
    diagnostic: dict[str, Any] = {
        "requested_degree": degree,
        "used_degree": None,
        "condition_number": float("nan"),
        "rank": 0,
        "stable": False,
        "fallback_reason": None,
        "fallback_level": float(levels[zero_index]),
    }
    if not fixed_support_identifiable:
        diagnostic["fallback_reason"] = "non_identifiable_fixed_support"
        return fallback, diagnostic
    if levels.size < degree + 1:
        diagnostic["fallback_reason"] = "insufficient_levels"
        return fallback, diagnostic

    vandermonde = np.vander(levels, N=degree + 1, increasing=False)
    rank = int(np.linalg.matrix_rank(vandermonde))
    condition = float(np.linalg.cond(vandermonde))
    diagnostic["rank"] = rank
    diagnostic["condition_number"] = condition
    if rank < degree + 1:
        diagnostic["fallback_reason"] = "rank_deficient_extrapolation"
        return fallback, diagnostic
    if not np.isfinite(condition) or condition > condition_limit:
        diagnostic["fallback_reason"] = "ill_conditioned_extrapolation"
        return fallback, diagnostic
    try:
        polynomial, _, _, _ = np.linalg.lstsq(vandermonde, path, rcond=None)
        at_minus_one = np.array([(-1.0) ** power for power in range(degree, -1, -1)])
        extrapolated = at_minus_one @ polynomial
    except np.linalg.LinAlgError:
        diagnostic["fallback_reason"] = "extrapolation_solver_failure"
        return fallback, diagnostic
    if not np.isfinite(extrapolated).all():
        diagnostic["fallback_reason"] = "non_finite_extrapolation"
        return fallback, diagnostic
    diagnostic["used_degree"] = degree
    diagnostic["stable"] = True
    return extrapolated, diagnostic


def fit_simex_fixed_support(
    levels: ArrayLike,
    reconstructions: Sequence[Sequence[RegressionSample]] | ReconstructionCallback,
    support: ArrayLike,
    *,
    n_replicates: int | None = None,
    extrapolation_degree: int = 2,
    extrapolation_condition_limit: float = 1.0e8,
) -> SIMEXRefitResult:
    """Refit a frozen support across SIMEX noise levels and extrapolate to ``-1``.

    ``reconstructions[level_index][replicate_index]`` supplies an already
    rebuilt ``(full_design_matrix, target)`` pair.  Alternatively, pass a
    callback taking ``(level, replicate_index)`` and specify ``n_replicates``.
    The function performs ordinary least squares only on ``support``; it never
    ranks, thresholds, expands, or otherwise changes that support.

    Extrapolation uses a polynomial through the mean coefficient path.  If the
    Vandermonde system is rank-deficient or exceeds
    ``extrapolation_condition_limit``, the returned coefficient is explicitly
    the uncorrected level-zero path value, with the
    reason recorded in ``diagnostics["extrapolation"]``.
    """
    if (
        not isinstance(extrapolation_degree, int)
        or isinstance(extrapolation_degree, bool)
        or extrapolation_degree < 1
    ):
        raise ValueError("extrapolation_degree must be an integer of at least one")
    if not np.isfinite(extrapolation_condition_limit) or extrapolation_condition_limit < 1.0:
        raise ValueError("extrapolation_condition_limit must be finite and at least one")

    checked_levels = _validate_levels(levels)
    checked_support = _validate_support(support)
    samples = _materialize_reconstructions(
        checked_levels, reconstructions, checked_support, n_replicates
    )
    replicates, path, errors, ranks, conditions = _fit_path(samples, checked_support)
    rank_deficient_replicates = tuple(
        np.flatnonzero(level_ranks < checked_support.size) for level_ranks in ranks
    )
    identifiable = not any(indices.size for indices in rank_deficient_replicates)
    extrapolated, extrapolation = _extrapolate_or_fallback(
        checked_levels,
        path,
        extrapolation_degree,
        extrapolation_condition_limit,
        identifiable,
    )
    diagnostics = {
        "replicate_counts": np.array([value.shape[0] for value in replicates], dtype=int),
        "ols_ranks": ranks,
        "ols_condition_numbers": conditions,
        "identifiability": {
            "identifiable": identifiable,
            "required_rank": int(checked_support.size),
            "rank_deficient_replicates": rank_deficient_replicates,
        },
        "extrapolation": extrapolation,
    }
    return SIMEXRefitResult(
        support=checked_support,
        levels=checked_levels,
        coefficient_path=path,
        coefficient_standard_errors=errors,
        replicate_coefficients=replicates,
        extrapolated_coefficients=extrapolated,
        diagnostics=diagnostics,
    )
