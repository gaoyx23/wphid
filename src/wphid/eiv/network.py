"""SIMEX paths for shared nonlinear mechanisms across multiple equations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from wphid.discovery.network_profiled import profiled_shared_mechanism_fit
from wphid.eiv.nonlinear import _fit_extrapolates, _path_statistics


NetworkReconstruction = Callable[
    [float, int],
    tuple[Sequence[np.ndarray], Sequence[np.ndarray], Callable[[np.ndarray], Sequence[np.ndarray]]],
]


@dataclass(frozen=True)
class NetworkSIMEXResult:
    levels: np.ndarray
    parameter_path: np.ndarray
    coefficient_path: np.ndarray
    level_zero_parameters: np.ndarray
    level_zero_coefficients: np.ndarray
    extrapolated_parameters: np.ndarray
    extrapolated_coefficients: np.ndarray
    parameter_path_spread: np.ndarray
    coefficient_path_spread: np.ndarray
    stable: bool
    diagnostics: dict


def fit_network_simex_path(
    levels: Sequence[float],
    reconstruction: NetworkReconstruction,
    active_equations: Sequence[bool],
    parameter_bounds: Sequence[tuple[float, float]],
    coefficient_bounds: Sequence[tuple[float, float]],
    *,
    n_replicates: int,
    seed: int,
    global_iterations: int = 30,
    variance_floor: float = 1e-10,
    variance_weighted: bool = True,
    maximum_relative_extrapolation: float = 0.50,
) -> NetworkSIMEXResult:
    """Rebuild and refit a shared-mechanism network along a SIMEX path."""
    noise_levels = np.asarray(levels, dtype=float)
    if (noise_levels.ndim != 1 or noise_levels.size < 3
            or noise_levels[0] != 0.0 or np.any(np.diff(noise_levels) <= 0.0)):
        raise ValueError("levels must be increasing, start at zero, and contain at least three values")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    replicate_parameters, replicate_coefficients = [], []
    fit_success = []
    coefficient_count = None
    for level_index, level in enumerate(noise_levels):
        local_parameters, local_coefficients = [], []
        for replicate in range(n_replicates):
            bases, targets, atom_builder = reconstruction(float(level), replicate)
            fit = profiled_shared_mechanism_fit(
                bases, targets, atom_builder, active_equations, parameter_bounds,
                seed=int(np.random.SeedSequence(
                    (seed, level_index, replicate, 0x4E4554)).generate_state(1)[0]),
                global_iterations=global_iterations,
            )
            flattened = np.concatenate(fit.coefficients)
            coefficient_count = flattened.size if coefficient_count is None else coefficient_count
            if flattened.size != coefficient_count:
                raise ValueError("network reconstruction changed coefficient dimension")
            local_parameters.append(fit.parameters)
            local_coefficients.append(flattened)
            fit_success.append(bool(
                np.all(np.isfinite(fit.parameters))
                and np.all(np.isfinite(flattened))
                and np.all(np.isfinite(fit.equation_rss))))
        replicate_parameters.append(np.vstack(local_parameters))
        replicate_coefficients.append(np.vstack(local_coefficients))
    if coefficient_count != len(coefficient_bounds):
        raise ValueError("coefficient_bounds must match flattened network coefficients")

    parameter_path, parameter_spread, parameter_inverse_variance = _path_statistics(
        replicate_parameters, variance_floor)
    coefficient_path, coefficient_spread, coefficient_inverse_variance = _path_statistics(
        replicate_coefficients, variance_floor)
    parameter_orders = np.ones(parameter_path.shape[1], dtype=int)
    coefficient_orders = np.ones(coefficient_path.shape[1], dtype=int)
    parameter_weights = (parameter_inverse_variance if variance_weighted
                         else np.ones_like(parameter_path))
    coefficient_weights = (coefficient_inverse_variance if variance_weighted
                           else np.ones_like(coefficient_path))
    extrapolated_parameters, parameter_feasible, parameter_conditions = _fit_extrapolates(
        noise_levels, parameter_path, parameter_weights,
        parameter_bounds, parameter_orders)
    extrapolated_coefficients, coefficient_feasible, coefficient_conditions = _fit_extrapolates(
        noise_levels, coefficient_path, coefficient_weights,
        coefficient_bounds, coefficient_orders)
    level_zero_parameters = parameter_path[0].copy()
    level_zero_coefficients = coefficient_path[0].copy()
    parameter_shift = np.abs(extrapolated_parameters-level_zero_parameters)/np.maximum(
        np.abs(level_zero_parameters), 1e-12)
    coefficient_shift = np.abs(extrapolated_coefficients-level_zero_coefficients)/np.maximum(
        np.abs(level_zero_coefficients), 1e-12)
    stable = bool(all(fit_success) and parameter_feasible and coefficient_feasible
                  and np.all(parameter_shift <= maximum_relative_extrapolation)
                  and np.all(coefficient_shift <= maximum_relative_extrapolation))
    if not stable:
        extrapolated_parameters = level_zero_parameters.copy()
        extrapolated_coefficients = level_zero_coefficients.copy()
    return NetworkSIMEXResult(
        levels=noise_levels, parameter_path=parameter_path,
        coefficient_path=coefficient_path,
        level_zero_parameters=level_zero_parameters,
        level_zero_coefficients=level_zero_coefficients,
        extrapolated_parameters=extrapolated_parameters,
        extrapolated_coefficients=extrapolated_coefficients,
        parameter_path_spread=parameter_spread,
        coefficient_path_spread=coefficient_spread, stable=stable,
        diagnostics={"parameter_weighted_design_condition_numbers": parameter_conditions,
                     "coefficient_weighted_design_condition_numbers": coefficient_conditions,
                     "variance_weighted": bool(variance_weighted),
                     "parameter_relative_extrapolation_shift": parameter_shift.tolist(),
                     "coefficient_relative_extrapolation_shift": coefficient_shift.tolist(),
                     "fallback_to_level_zero": not stable,
                     "all_network_fits_numerically_valid": bool(all(fit_success))},
    )
