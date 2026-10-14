"""Projected local derivatives with explicit calibration and source loading.

No truth, trajectory, noise draws, optimizer, or scientific precision gate is
used here. Derivatives must include all state-dependent loading terms. A fixed
whitener specifies a metric; it is not automatically an optimal covariance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InformationDiagnostic:
    information: np.ndarray
    residual_derivative: np.ndarray
    singular_values: np.ndarray
    parameter_null_basis: np.ndarray
    rank: int
    n_parameters: int
    nuisance_rank: int
    rank_tolerance: float
    smallest_singular_value: float
    condition_ratio: float


def _matrix(value, name, rows=None):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} has the wrong row count")
    return array


def _relative_tolerance(rtol):
    if not np.isfinite(rtol) or not 0 < rtol < .1:
        raise ValueError("rtol must be finite and between zero and .1")
    return float(rtol)


def _whitener(value, rows):
    if value is None:
        return None
    matrix = _matrix(value, "whitener", rows)
    if matrix.shape[1] != rows:
        raise ValueError("whitener must be square")
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    if spectrum[-1] <= np.finfo(float).eps * rows * spectrum[0]:
        raise ValueError("whitener must be numerically invertible")
    return matrix


def _span(matrix, rtol):
    # Nuisance parameter units must not determine the column space.
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0))
    maximum = np.max(np.abs(matrix), axis=0)
    active = maximum > 0
    if not np.any(active):
        return np.empty((matrix.shape[0], 0))
    scaled = matrix[:, active] / maximum[active]
    scaled /= np.linalg.norm(scaled, axis=0)
    left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.count_nonzero(singular > rtol * singular[0]))
    return left[:, :rank]


def _diagnose(derivative, nuisance, rtol):
    basis = _span(nuisance, rtol)
    residual = derivative - basis @ (basis.T @ derivative)
    # Retain the parameter null space without allocating an observation-sized U.
    _, singular, right = np.linalg.svd(
        residual, full_matrices=residual.shape[0] < residual.shape[1])
    n_parameters = derivative.shape[1]
    spectrum = np.zeros(n_parameters)
    spectrum[:len(singular)] = singular
    reference = float(np.linalg.norm(derivative, ord=2))
    tolerance = rtol * reference if reference else rtol
    rank = int(np.count_nonzero(spectrum > tolerance))
    information = residual.T @ residual
    information = .5 * (information + information.T)
    largest = float(spectrum[0])
    smallest = float(spectrum[-1])
    return InformationDiagnostic(
        information=information, residual_derivative=residual,
        singular_values=spectrum, parameter_null_basis=right[rank:].T,
        rank=rank, n_parameters=n_parameters, nuisance_rank=basis.shape[1],
        rank_tolerance=tolerance, smallest_singular_value=smallest,
        condition_ratio=smallest / largest if largest else 0.,
    )


def profile_information(kinetic_derivative, nuisance_derivative, *,
                        whitener=None, parameter_scale=None, rtol=1e-10):
    """Project kinetic derivatives off the complete local nuisance space.

    parameter_scale contains positive physical units for the kinetic unknowns.
    Zero columns and rank failure are retained, without ridge stabilization.
    This produces local geometry, not parameter estimates or confidence bounds.
    """
    rtol = _relative_tolerance(rtol)
    derivative = _matrix(kinetic_derivative, "kinetic_derivative")
    rows, parameters = derivative.shape
    if rows == 0 or parameters == 0:
        raise ValueError("at least one observation row and parameter are required")
    nuisance = _matrix(nuisance_derivative, "nuisance_derivative", rows)
    scale = np.ones(parameters) if parameter_scale is None else np.asarray(
        parameter_scale, dtype=float)
    if scale.shape != (parameters,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("parameter_scale must contain positive finite units")
    metric = _whitener(whitener, rows)
    scaled = derivative * scale
    return _diagnose(scaled if metric is None else metric @ scaled,
                     nuisance if metric is None else metric @ nuisance, rtol)


def independent_calibration_information(
        kinetic_derivative, nuisance_derivative, calibration_derivative,
        calibration_precision_factor, *, whitener=None, parameter_scale=None,
        rtol=1e-10):
    """Add independent calibration observation rows before nuisance projection.

    calibration_precision_factor is the whitened calibration-data derivative
    with respect to the same calibration unknowns. Zero rows mean no independent
    calibration information. This interface does not model correlated assays or
    convert an arbitrary regularizer into calibration evidence.
    """
    derivative = _matrix(kinetic_derivative, "kinetic_derivative")
    rows, parameters = derivative.shape
    if rows == 0 or parameters == 0:
        raise ValueError("at least one observation row and parameter are required")
    nuisance = _matrix(nuisance_derivative, "nuisance_derivative", rows)
    calibration = _matrix(calibration_derivative, "calibration_derivative", rows)
    precision = _matrix(calibration_precision_factor, "calibration_precision_factor")
    if precision.shape[1] != calibration.shape[1]:
        raise ValueError("calibration precision has the wrong column count")
    metric = _whitener(whitener, rows)
    whitened_derivative = derivative if metric is None else metric @ derivative
    whitened_nuisance = nuisance if metric is None else metric @ nuisance
    whitened_calibration = calibration if metric is None else metric @ calibration
    augmented_derivative = np.vstack((whitened_derivative,
                                      np.zeros((precision.shape[0], parameters))))
    augmented_nuisance = np.block([
        [whitened_nuisance, whitened_calibration],
        [np.zeros((precision.shape[0], nuisance.shape[1])), precision],
    ])
    return profile_information(augmented_derivative, augmented_nuisance,
                               parameter_scale=parameter_scale, rtol=rtol)


def nested_nuisance_spaces(free_nuisance, restricted_nuisance, *,
                           whitener=None, rtol=1e-10):
    """Check a necessary premise of the same-derivative information inequality.

    True does not establish correct physics, identical kinetic derivatives,
    statistical efficiency, or a scientific accuracy improvement.
    """
    rtol = _relative_tolerance(rtol)
    free = _matrix(free_nuisance, "free_nuisance")
    if free.shape[0] == 0:
        raise ValueError("at least one row is required")
    restricted = _matrix(restricted_nuisance, "restricted_nuisance", free.shape[0])
    metric = _whitener(whitener, free.shape[0])
    free_basis = _span(free if metric is None else metric @ free, rtol)
    restricted_basis = _span(restricted if metric is None else metric @ restricted, rtol)
    if restricted_basis.shape[1] == 0:
        return True
    outside = restricted_basis - free_basis @ (free_basis.T @ restricted_basis)
    return bool(np.linalg.norm(outside, ord=2) <= 10 * rtol)


def weak_source_columns(weak_weights, rate_shapes, loadings):
    """Integrate loading*rate per node; do not pull state loading outside.

    Shapes: weights=(fields, windows, nodes), rate_shapes=(rates, nodes),
    loadings=(fields, rates, nodes). Rows are field-major then window-major.
    Quadrature factors and weak tests must already be in weak_weights. This
    handles source columns only, not complete PDE balance or noise correction.
    """
    weights = np.asarray(weak_weights, dtype=float)
    rates = _matrix(rate_shapes, "rate_shapes")
    loading = np.asarray(loadings, dtype=float)
    if weights.ndim != 3 or loading.ndim != 3:
        raise ValueError("weights and loadings must be three-dimensional")
    fields, windows, nodes = weights.shape
    if min(fields, windows, nodes, rates.shape[0]) == 0:
        raise ValueError("fields, windows, nodes, and rates must be nonempty")
    if rates.shape[1] != nodes or loading.shape != (fields, rates.shape[0], nodes):
        raise ValueError("source loading shapes do not match weights and rates")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(loading)):
        raise ValueError("weights and loadings must be finite")
    return np.einsum("ewn,ern,rn->ewr", weights, loading, rates).reshape(
        fields * windows, rates.shape[0])
