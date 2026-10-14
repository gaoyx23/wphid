"""Exact best-subset selection by information criteria (AIC / BIC)."""

from collections.abc import Callable

import numpy as np


_MAX_ENUMERATION_COLUMNS = 20


def _best_subset_enumeration(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: Callable[[int], float],
) -> tuple[np.ndarray, float]:
    if matrix.ndim != 2 or target.ndim != 1 or matrix.shape[0] != target.shape[0]:
        raise ValueError("A must be two-dimensional and b one-dimensional with matching rows")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("A and b must contain only finite values")
    n, p = matrix.shape
    if p > _MAX_ENUMERATION_COLUMNS:
        raise ValueError(
            f"exact best-subset enumeration supports at most "
            f"{_MAX_ENUMERATION_COLUMNS} columns, got {p}"
        )

    best_key: tuple[float, int, tuple[int, ...]] | None = None
    best_support: np.ndarray | None = None
    for mask in range(1 << p):
        support = np.array([(mask >> j) & 1 for j in range(p)], dtype=int)
        indices = np.flatnonzero(support)
        k = int(indices.size)
        if k == 0:
            rss = float(np.dot(target, target))
        else:
            if np.linalg.matrix_rank(matrix[:, indices]) < k:
                continue
            coefficients = np.linalg.lstsq(matrix[:, indices], target, rcond=None)[0]
            residual = matrix[:, indices] @ coefficients - target
            rss = float(np.dot(residual, residual))
        rss = max(rss, 1e-300)
        score = float(n * np.log(rss / n) + penalty(k))
        key = (score, k, tuple(int(i) for i in indices))
        if best_key is None or key < best_key:
            best_key = key
            best_support = indices
    if best_key is None or best_support is None:
        raise ValueError("best-subset selection produced no candidate support")
    return best_support, best_key[0]


def best_subset_aic(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the AIC-optimal exact support of ``A x = b``."""
    matrix = np.asarray(A, dtype=float)
    target = np.asarray(b, dtype=float)
    return _best_subset_enumeration(matrix, target, penalty=lambda k: 2.0 * k)


def best_subset_bic(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the BIC-optimal exact support of ``A x = b``."""
    matrix = np.asarray(A, dtype=float)
    target = np.asarray(b, dtype=float)
    n = int(matrix.shape[0])
    return _best_subset_enumeration(
        matrix, target, penalty=lambda k: float(k) * np.log(n)
    )
