"""Frozen-family nonlinear SIMEX paths with full design reconstruction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from wphid.discovery.profiled import ProfiledFit, profiled_nonlinear_fit


Reconstruction = Callable[[float, int], tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]]


@dataclass(frozen=True)
class NonlinearSIMEXResult:
    levels: np.ndarray
    parameter_path: np.ndarray
    parameter_standard_deviation: np.ndarray
    replicate_parameters: tuple[np.ndarray, ...]
    level_zero_parameters: np.ndarray
    extrapolated_parameters: np.ndarray
    stable: bool
    diagnostics: dict
    extrapolated_standard_errors: np.ndarray | None = None
    extrapolated_covariance: np.ndarray | None = None
    bootstrap_extrapolated_parameters: np.ndarray | None = None
    selected_path_orders: np.ndarray | None = None
    coefficient_path: np.ndarray | None = None
    coefficient_standard_deviation: np.ndarray | None = None
    replicate_coefficients: tuple[np.ndarray, ...] | None = None
    level_zero_coefficients: np.ndarray | None = None
    extrapolated_coefficients: np.ndarray | None = None
    coefficient_standard_errors: np.ndarray | None = None
    coefficient_covariance: np.ndarray | None = None
    bootstrap_extrapolated_coefficients: np.ndarray | None = None
    selected_coefficient_path_orders: np.ndarray | None = None
    bootstrap_joint_estimates: np.ndarray | None = None
    joint_covariance: np.ndarray | None = None


def _path_design(levels: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(levels, N=degree + 1, increasing=True)


def _bounded_weighted_path(
    levels: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    degree: int,
    bound: tuple[float, float],
) -> tuple[np.ndarray, float, bool]:
    """Fit a weighted polynomial whose observed path and extrapolate obey bounds."""
    design = _path_design(levels, degree)
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    weighted_values = values * root_weights
    initial = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
    constrained_design = _path_design(np.r_[-1.0, levels], degree)
    constraint = LinearConstraint(constrained_design, bound[0], bound[1])

    def objective(coefficients: np.ndarray) -> float:
        residual = weighted_design @ coefficients - weighted_values
        return 0.5 * float(residual @ residual)

    def gradient(coefficients: np.ndarray) -> np.ndarray:
        return weighted_design.T @ (weighted_design @ coefficients - weighted_values)

    initial_predictions = constrained_design @ initial
    initial_feasible = bool(
        np.all(initial_predictions >= bound[0])
        and np.all(initial_predictions <= bound[1])
    )
    fitted = None
    if initial_feasible:
        coefficients = initial
    else:
        fitted = minimize(
            objective,
            initial,
            jac=gradient,
            constraints=(constraint,),
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        coefficients = fitted.x if fitted.success else initial
    extrapolated = float((_path_design(np.asarray([-1.0]), degree) @ coefficients).item())
    feasible = bool(
        (initial_feasible or (fitted is not None and fitted.success))
        and bound[0] - 1e-8 <= extrapolated <= bound[1] + 1e-8
        and np.all(constrained_design @ coefficients >= bound[0] - 1e-8)
        and np.all(constrained_design @ coefficients <= bound[1] + 1e-8)
    )
    return coefficients, float(np.clip(extrapolated, *bound)), feasible


def _select_path_order(
    levels: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    candidates: Sequence[int],
    bound: tuple[float, float],
) -> tuple[int, dict[int, float]]:
    """Select path order by truth-free leave-one-level-out prediction error."""
    scores: dict[int, float] = {}
    for degree in candidates:
        errors = []
        for held_out in range(levels.size):
            keep = np.arange(levels.size) != held_out
            if np.count_nonzero(keep) <= degree:
                errors.append(np.inf)
                continue
            coefficients, _, feasible = _bounded_weighted_path(
                levels[keep], values[keep], weights[keep], degree, bound
            )
            prediction = float((_path_design(levels[[held_out]], degree) @ coefficients).item())
            error = weights[held_out] * (prediction - values[held_out]) ** 2
            errors.append(error if feasible else np.inf)
        scores[int(degree)] = float(np.mean(errors))
    selected = min(scores, key=lambda degree: (scores[degree], degree))
    return int(selected), scores


def _path_statistics(
    replicate_values: Sequence[np.ndarray], variance_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.vstack([np.mean(values, axis=0) for values in replicate_values])
    spread = np.vstack([np.std(values, axis=0, ddof=0) for values in replicate_values])
    counts = np.asarray([values.shape[0] for values in replicate_values], dtype=float)[:, None]
    variance_of_mean = np.maximum(spread**2 / counts, variance_floor)
    return path, spread, 1.0 / variance_of_mean


def _fit_extrapolates(
    levels: np.ndarray,
    path: np.ndarray,
    weights: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    orders: np.ndarray,
) -> tuple[np.ndarray, bool, list[float]]:
    extrapolated = np.empty(path.shape[1])
    feasible = True
    conditions = []
    for index, degree in enumerate(orders):
        design = _path_design(levels, int(degree))
        conditions.append(float(np.linalg.cond(design * np.sqrt(weights[:, index])[:, None])))
        _, extrapolated[index], fit_feasible = _bounded_weighted_path(
            levels, path[:, index], weights[:, index], int(degree), bounds[index]
        )
        feasible = feasible and fit_feasible
    return extrapolated, feasible, conditions


def fit_nonlinear_simex_path(
    levels: Sequence[float],
    reconstruction: Reconstruction,
    bounds: Sequence[tuple[float, float]],
    *,
    n_replicates: int,
    seed: int,
    extrapolation_degree: int = 2,
    maximum_relative_extrapolation: float = 0.35,
    variance_weighted: bool = False,
    path_order_candidates: Sequence[int] | None = None,
    bootstrap_replicates: int = 0,
    variance_floor: float = 1e-10,
    coefficient_bounds: Sequence[tuple[float, float]] | None = None,
    coefficient_path_order_candidates: Sequence[int] | None = None,
) -> NonlinearSIMEXResult:
    """Rebuild and refit a nonlinear family at every SIMEX noise level.

    ``path_order_candidates`` enables truth-free leave-one-level-out selection.
    Passing a single candidate freezes that order. ``variance_weighted`` uses the
    replicate variance of each level mean, and ``bootstrap_replicates`` resamples
    replicate-level estimates within levels to quantify extrapolation uncertainty.
    The defaults preserve the historical fixed-quadratic, unweighted call contract.
    """
    noise_levels = np.asarray(levels, dtype=float)
    if (noise_levels.ndim != 1 or noise_levels.size < 3
            or not np.all(np.diff(noise_levels) > 0.0)
            or noise_levels[0] != 0.0 or np.any(noise_levels < 0.0)):
        raise ValueError("levels must be increasing, start at zero, and contain at least three values")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    candidates = tuple(path_order_candidates or (extrapolation_degree,))
    if not candidates or any(degree not in (1, 2) for degree in candidates):
        raise ValueError("path orders must be one or both of 1 and 2")
    if any(noise_levels.size <= degree for degree in candidates):
        raise ValueError("extrapolation degree must be supported by the level count")

    replicate_values = []
    replicate_coefficients = []
    success = []
    for level_index, level in enumerate(noise_levels):
        fitted = []
        fitted_coefficients = []
        for replicate in range(n_replicates):
            base, target, atom_builder = reconstruction(float(level), replicate)
            fit: ProfiledFit = profiled_nonlinear_fit(
                base, target, atom_builder, bounds,
                seed=int(np.random.SeedSequence((seed, level_index, replicate)).generate_state(1)[0]),
                global_iterations=35,
            )
            fitted.append(fit.parameters)
            fitted_coefficients.append(fit.coefficients)
            success.append(fit.success)
        replicate_values.append(np.vstack(fitted))
        replicate_coefficients.append(np.vstack(fitted_coefficients))

    coefficient_count = replicate_coefficients[0].shape[1]
    if any(values.shape[1] != coefficient_count for values in replicate_coefficients):
        raise ValueError("reconstruction must produce a fixed linear coefficient count")
    if coefficient_bounds is None:
        coefficient_bounds = [(-np.inf, np.inf)]*coefficient_count
    if len(coefficient_bounds) != coefficient_count:
        raise ValueError("coefficient_bounds must match the profiled coefficient count")
    coefficient_candidates = tuple(coefficient_path_order_candidates or candidates)
    if (not coefficient_candidates
            or any(degree not in (1, 2) for degree in coefficient_candidates)):
        raise ValueError("coefficient path orders must be one or both of 1 and 2")

    path, spread, inverse_variances = _path_statistics(replicate_values, variance_floor)
    weights = inverse_variances if variance_weighted else np.ones_like(path)
    selected_orders = np.empty(path.shape[1], dtype=int)
    selection_scores = []
    for index in range(path.shape[1]):
        selected_orders[index], scores = _select_path_order(
            noise_levels, path[:, index], weights[:, index], candidates, bounds[index]
        )
        selection_scores.append(scores)

    extrapolated, path_feasible, conditions = _fit_extrapolates(
        noise_levels, path, weights, bounds, selected_orders
    )
    geometry_conditions = [float(np.linalg.cond(_path_design(noise_levels, int(degree))))
                           for degree in selected_orders]
    level_zero = path[0].copy()
    relative_shift = np.abs(extrapolated-level_zero)/np.maximum(np.abs(level_zero), 1e-12)
    stable = bool(all(success) and path_feasible and np.all(np.isfinite(conditions))
                  and np.all(np.isfinite(geometry_conditions))
                  and max(geometry_conditions) < 1.0e4
                  and np.all(relative_shift <= maximum_relative_extrapolation))
    if not stable:
        extrapolated = level_zero.copy()

    coefficient_path, coefficient_spread, coefficient_inverse_variances = _path_statistics(
        replicate_coefficients, variance_floor)
    coefficient_weights = (coefficient_inverse_variances if variance_weighted
                           else np.ones_like(coefficient_path))
    coefficient_orders = np.empty(coefficient_count, dtype=int)
    coefficient_selection_scores = []
    for index in range(coefficient_count):
        coefficient_orders[index], scores = _select_path_order(
            noise_levels, coefficient_path[:, index], coefficient_weights[:, index],
            coefficient_candidates, coefficient_bounds[index])
        coefficient_selection_scores.append(scores)
    extrapolated_coefficients, coefficient_path_feasible, coefficient_conditions = (
        _fit_extrapolates(noise_levels, coefficient_path, coefficient_weights,
                          coefficient_bounds, coefficient_orders))
    level_zero_coefficients = coefficient_path[0].copy()
    if not coefficient_path_feasible:
        extrapolated_coefficients = level_zero_coefficients.copy()

    bootstrap_samples = None
    covariance = None
    standard_errors = None
    coefficient_bootstrap_samples = None
    coefficient_covariance = None
    coefficient_standard_errors = None
    joint_bootstrap_samples = None
    joint_covariance = None
    if bootstrap_replicates:
        rng = np.random.default_rng(np.random.SeedSequence((seed, 0xB0057)))
        samples = []
        coefficient_samples = []
        for _ in range(bootstrap_replicates):
            indices = [rng.integers(0, values.shape[0], size=values.shape[0])
                       for values in replicate_values]
            resampled = [values[index] for values, index in zip(replicate_values, indices)]
            resampled_coefficients = [values[index] for values, index
                                      in zip(replicate_coefficients, indices)]
            bootstrap_path, _, bootstrap_inverse_variances = _path_statistics(
                resampled, variance_floor
            )
            bootstrap_weights = (
                bootstrap_inverse_variances if variance_weighted else np.ones_like(bootstrap_path)
            )
            sample, feasible, _ = _fit_extrapolates(
                noise_levels, bootstrap_path, bootstrap_weights, bounds, selected_orders
            )
            coefficient_bootstrap_path, _, coefficient_bootstrap_inverse_variances = (
                _path_statistics(resampled_coefficients, variance_floor))
            coefficient_bootstrap_weights = (
                coefficient_bootstrap_inverse_variances if variance_weighted
                else np.ones_like(coefficient_bootstrap_path))
            coefficient_sample, coefficient_feasible, _ = _fit_extrapolates(
                noise_levels, coefficient_bootstrap_path, coefficient_bootstrap_weights,
                coefficient_bounds, coefficient_orders)
            joint_feasible = (feasible and coefficient_feasible
                              and np.all(np.isfinite(sample))
                              and np.all(np.isfinite(coefficient_sample)))
            if joint_feasible:
                samples.append(sample)
                coefficient_samples.append(coefficient_sample)
        bootstrap_samples = np.asarray(samples, dtype=float).reshape(-1, path.shape[1])
        if bootstrap_samples.shape[0] >= 2:
            covariance = np.atleast_2d(np.cov(bootstrap_samples, rowvar=False, ddof=1))
            standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        else:
            covariance = np.full((path.shape[1], path.shape[1]), np.nan)
            standard_errors = np.full(path.shape[1], np.nan)
        coefficient_bootstrap_samples = np.asarray(
            coefficient_samples, dtype=float).reshape(-1, coefficient_count)
        if coefficient_bootstrap_samples.shape[0] >= 2:
            coefficient_covariance = np.atleast_2d(
                np.cov(coefficient_bootstrap_samples, rowvar=False, ddof=1))
            coefficient_standard_errors = np.sqrt(
                np.maximum(np.diag(coefficient_covariance), 0.0))
        else:
            coefficient_covariance = np.full((coefficient_count, coefficient_count), np.nan)
            coefficient_standard_errors = np.full(coefficient_count, np.nan)
        if (bootstrap_samples.shape[0] >= 2
                and bootstrap_samples.shape[0] == coefficient_bootstrap_samples.shape[0]):
            # Preserve parameter--coefficient compensation (especially theta--A)
            # by forming the covariance from paired bootstrap reconstructions.
            joint_bootstrap_samples = np.column_stack(
                (coefficient_bootstrap_samples, bootstrap_samples))
            joint_covariance = np.atleast_2d(
                np.cov(joint_bootstrap_samples, rowvar=False, ddof=1))

    return NonlinearSIMEXResult(
        levels=noise_levels,
        parameter_path=path,
        parameter_standard_deviation=spread,
        replicate_parameters=tuple(replicate_values),
        level_zero_parameters=level_zero,
        extrapolated_parameters=extrapolated,
        stable=stable,
        diagnostics={
            "extrapolation_degree": extrapolation_degree,
            "selected_path_orders": selected_orders.tolist(),
            "path_order_selection_scores": selection_scores,
            "variance_weighted": variance_weighted,
            "variance_floor": variance_floor,
            "weighted_design_condition_numbers": conditions,
            "path_geometry_condition_numbers": geometry_conditions,
            "relative_extrapolation_shift": relative_shift.tolist(),
            "fallback_to_level_zero": not stable,
            "bootstrap_requested": bootstrap_replicates,
            "bootstrap_successful": 0 if bootstrap_samples is None else bootstrap_samples.shape[0],
            "coefficient_path_order_selection_scores": coefficient_selection_scores,
            "selected_coefficient_path_orders": coefficient_orders.tolist(),
            "coefficient_weighted_design_condition_numbers": coefficient_conditions,
            "coefficient_path_feasible": coefficient_path_feasible,
            "coefficient_bootstrap_successful": (
                0 if coefficient_bootstrap_samples is None
                else coefficient_bootstrap_samples.shape[0]),
            "joint_bootstrap_successful": (
                0 if joint_bootstrap_samples is None else joint_bootstrap_samples.shape[0]),
        },
        extrapolated_standard_errors=standard_errors,
        extrapolated_covariance=covariance,
        bootstrap_extrapolated_parameters=bootstrap_samples,
        selected_path_orders=selected_orders,
        coefficient_path=coefficient_path,
        coefficient_standard_deviation=coefficient_spread,
        replicate_coefficients=tuple(replicate_coefficients),
        level_zero_coefficients=level_zero_coefficients,
        extrapolated_coefficients=extrapolated_coefficients,
        coefficient_standard_errors=coefficient_standard_errors,
        coefficient_covariance=coefficient_covariance,
        bootstrap_extrapolated_coefficients=coefficient_bootstrap_samples,
        selected_coefficient_path_orders=coefficient_orders,
        bootstrap_joint_estimates=joint_bootstrap_samples,
        joint_covariance=joint_covariance,
    )
