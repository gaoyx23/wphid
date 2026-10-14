"""Profiled nonlinear regression for sparse physics discovery.

Nonlinear parameters define a candidate atom while its amplitude and all base
terms are solved by linear least squares.  This avoids conflating amplitude
scaling with activation-energy or exponent estimation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize


AtomBuilder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ProfiledFit:
    parameters: np.ndarray
    coefficients: np.ndarray
    rss: float
    bic: float
    success: bool


def _linear_fit(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    residual = target - matrix @ coefficients
    return coefficients, max(float(residual @ residual), np.finfo(float).tiny)


def profiled_nonlinear_fit(
    base_matrix: np.ndarray,
    target: np.ndarray,
    atom_builder: AtomBuilder,
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int = 0,
    global_iterations: int = 60,
) -> ProfiledFit:
    """Fit nonlinear atom parameters with linear coefficients profiled out."""
    base = np.asarray(base_matrix, dtype=float)
    response = np.asarray(target, dtype=float)
    if base.ndim != 2 or response.ndim != 1 or base.shape[0] != response.size:
        raise ValueError("base_matrix and target must have matching rows")
    if not bounds or any(low >= high for low, high in bounds):
        raise ValueError("each nonlinear parameter bound must satisfy low < high")

    def objective(parameters: np.ndarray) -> float:
        atom = np.asarray(atom_builder(np.asarray(parameters, dtype=float)), dtype=float)
        if atom.shape != response.shape or not np.isfinite(atom).all():
            return float("inf")
        _, rss = _linear_fit(np.column_stack((base, atom)), response)
        return float(np.log(rss))

    global_result = differential_evolution(
        objective, list(bounds), seed=int(seed), polish=False,
        maxiter=int(global_iterations), popsize=10, tol=1e-8, workers=1,
    )
    local_result = minimize(
        objective, global_result.x, method="Nelder-Mead", bounds=list(bounds),
        options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-10},
    )
    parameters = np.asarray(local_result.x if local_result.success else global_result.x)
    parameters = np.clip(parameters,
                         [bound[0] for bound in bounds],
                         [bound[1] for bound in bounds])
    atom = np.asarray(atom_builder(parameters), dtype=float)
    coefficients, rss = _linear_fit(np.column_stack((base, atom)), response)
    n = response.size
    parameter_count = base.shape[1] + 1 + len(bounds)
    bic = float(n * np.log(rss / n) + parameter_count * np.log(n))
    return ProfiledFit(parameters, coefficients, rss, bic,
                       bool(global_result.success or local_result.success))


def fixed_parameter_profile(
    base_matrix: np.ndarray,
    target: np.ndarray,
    atom_builder: AtomBuilder,
    parameter_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RSS and profiled atom amplitude over a scalar parameter grid."""
    base = np.asarray(base_matrix, dtype=float)
    response = np.asarray(target, dtype=float)
    values = np.asarray(parameter_values, dtype=float)
    rss = np.empty(values.size)
    amplitude = np.empty(values.size)
    for index, value in enumerate(values):
        atom = np.asarray(atom_builder(np.array([value])), dtype=float)
        coefficients, rss[index] = _linear_fit(np.column_stack((base, atom)), response)
        amplitude[index] = coefficients[-1]
    return rss, amplitude
