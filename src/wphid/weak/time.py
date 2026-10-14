"""Compact-support temporal weak-form projections."""

import numpy as np


def project_time_weak(
    t: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a scalar time series onto compact-support weak test functions.

    For every complete sample window, this returns one row of ``A`` and one
    entry of ``b`` such that ``A @ coefficients = b`` in the weak form.
    """
    time, values, terms = _validate_inputs(t, y, features, window, stride)
    starts = range(0, time.size - window + 1, stride)
    rows = []
    targets = []

    for start in starts:
        stop = start + window
        local_time = time[start:stop]
        psi, psi_t = _polynomial_bump(local_time)
        rows.append(np.trapezoid(psi[:, None] * terms[start:stop], local_time, axis=0))
        targets.append(-np.trapezoid(psi_t * values[start:stop], local_time))

    return np.asarray(rows), np.asarray(targets)


def _polynomial_bump(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (time[0] + time[-1])
    half_width = 0.5 * (time[-1] - time[0])
    coordinate = (time - center) / half_width
    one_minus_square = 1.0 - coordinate**2
    psi = one_minus_square**4
    psi_t = -8.0 * coordinate * one_minus_square**3 / half_width
    return psi, psi_t


def _validate_inputs(
    t: np.ndarray, y: np.ndarray, features: np.ndarray, window: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(t, dtype=float)
    values = np.asarray(y, dtype=float)
    terms = np.asarray(features, dtype=float)
    if time.ndim != 1 or time.size < 3 or not np.all(np.diff(time) > 0.0):
        raise ValueError("t must be a strictly increasing one-dimensional array")
    if values.ndim != 1 or values.shape != time.shape:
        raise ValueError("y must be one-dimensional and match t")
    if terms.ndim != 2 or terms.shape[0] != time.size:
        raise ValueError("features must have shape (n_time, n_terms)")
    if not isinstance(window, (int, np.integer)) or not 3 <= window <= time.size:
        raise ValueError("window must be an integer from 3 through len(t)")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if not (np.all(np.isfinite(time)) and np.all(np.isfinite(values)) and np.all(np.isfinite(terms))):
        raise ValueError("t, y, and features must be finite")
    return time, values, terms
