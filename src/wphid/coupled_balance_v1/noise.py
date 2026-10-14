"""Shared-observation first-order noise propagation; no noise calibration.

Sparse coordinate Jacobians preserve target/design cross terms. Noise factors
are supplied assumptions or independent calibration inputs, never inferred here.
No bias correction, reconstruction model, clipping approximation or confidence
claim follows from this linearization.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, hstack, issparse


class CovarianceRankFailure(ValueError):
    pass


def _sparse_matrix(value, shape, name):
    result = csr_matrix(value)
    if result.shape != shape or not np.all(np.isfinite(result.data)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class SharedObservationLinearization:
    target_jacobian: object
    column_jacobians: tuple[object, ...]
    coordinate_keys: tuple[str, ...]

    def __post_init__(self):
        keys = self.coordinate_keys
        if not keys or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("raw observation coordinates must be unique and explicitly named")
        shape = self.target_jacobian.shape if hasattr(self.target_jacobian, "shape") else np.asarray(self.target_jacobian).shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] != len(keys):
            raise ValueError("target Jacobian has the wrong observation dimension")
        object.__setattr__(self, "target_jacobian", _sparse_matrix(self.target_jacobian, shape, "target Jacobian"))
        object.__setattr__(self, "column_jacobians", tuple(
            _sparse_matrix(value, shape, "design column Jacobian") for value in self.column_jacobians))

    def residual_jacobian(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (len(self.column_jacobians),) or not np.all(np.isfinite(coefficients)):
            raise ValueError("linear coefficients have the wrong shape or nonfinite entries")
        result = self.target_jacobian.copy()
        for amplitude, column in zip(coefficients, self.column_jacobians):
            if amplitude:
                result -= amplitude*column
        result.eliminate_zeros()
        return result

    def noise_operator(self, coefficients, observation_factor):
        return ResidualNoiseOperator(self.residual_jacobian(coefficients), observation_factor)


def observation_factor(diagonal_variance, shared_factors=None):
    """Construct F for Sigma_y=F F^T from supplied variances and shared errors."""
    variance = np.asarray(diagonal_variance, dtype=float)
    if variance.ndim != 1 or len(variance) == 0 or not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("observation variances must be finite and nonnegative")
    factor = diags(np.sqrt(variance), format="csr")
    if shared_factors is not None:
        shared = csr_matrix(shared_factors)
        if shared.shape[0] != len(variance) or not np.all(np.isfinite(shared.data)):
            raise ValueError("shared error factors have the wrong shape or nonfinite entries")
        factor = hstack((factor, shared), format="csr")
    return factor


class ResidualNoiseOperator:
    def __init__(self, residual_jacobian, factor):
        jacobian = csr_matrix(residual_jacobian)
        factor = csr_matrix(factor)
        if (jacobian.ndim != 2 or jacobian.shape[0] < 1 or factor.shape[0] != jacobian.shape[1]
                or not np.all(np.isfinite(jacobian.data)) or not np.all(np.isfinite(factor.data))):
            raise ValueError("residual Jacobian and noise factor have incompatible shapes or nonfinite entries")
        self.root = jacobian @ factor
        self.shape = (jacobian.shape[0], jacobian.shape[0])

    def matvec(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.shape[1],) or not np.all(np.isfinite(vector)):
            raise ValueError("covariance vector has the wrong shape or nonfinite entries")
        return np.asarray(self.root @ (self.root.T @ vector)).ravel()

    def diagonal(self):
        return np.asarray(self.root.multiply(self.root).sum(axis=1)).ravel()

    def dense(self, *, max_rows=512):
        if self.shape[0] > max_rows:
            raise ValueError("dense covariance cap exceeded; use covariance actions or qualified blocks")
        covariance = (self.root @ self.root.T).toarray()
        return .5*(covariance+covariance.T)

    def whitener(self, *, max_rows=512, rtol=1e-10):
        if not np.isfinite(rtol) or not 0 < rtol < .1:
            raise ValueError("rank tolerance must be finite and between zero and .1")
        covariance = self.dense(max_rows=max_rows)
        spectrum = np.linalg.eigvalsh(covariance)
        if spectrum[-1] <= 0 or spectrum[0] <= rtol*spectrum[-1]:
            raise CovarianceRankFailure("covariance is singular or numerically unresolved; no jitter was added")
        lower = np.linalg.cholesky(covariance)
        return np.linalg.solve(lower, np.eye(len(covariance)))


def numerical_balance_linearization(mapper, observation, coordinate_keys, steps, *, max_coordinates=256):
    """Small-input verification helper for a fully recomputed balance mapper.

    mapper returns a WeakSystem and must recompute rate/modulation/boundary terms
    from every perturbed raw coordinate. Holding a data-dependent modulation fixed
    omits its noise contribution. Domain failure propagates, never gets clipped.
    Production-size arrays need qualified analytic/sparse Jacobian assembly.
    """
    observation = np.asarray(observation, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if (observation.ndim != 1 or len(observation) != len(coordinate_keys) or
            len(observation) > max_coordinates or not np.all(np.isfinite(observation))):
        raise ValueError("raw coordinates are incompatible or exceed the verification cap")
    if steps.shape != observation.shape or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("finite positive derivative steps must be explicit")
    if len(set(coordinate_keys)) != len(coordinate_keys):
        raise ValueError("raw coordinates must be unique")
    baseline = mapper(observation.copy())
    rows, columns = baseline.matrix.shape
    target_jacobian = np.empty((rows, len(observation)))
    design_jacobian = np.empty((columns, rows, len(observation)))
    for index, step in enumerate(steps):
        upper, lower = observation.copy(), observation.copy()
        upper[index] += step
        lower[index] -= step
        plus, minus = mapper(upper), mapper(lower)
        for evaluated in (plus, minus):
            if (evaluated.matrix.shape != baseline.matrix.shape or evaluated.target.shape != baseline.target.shape
                    or evaluated.row_keys != baseline.row_keys or evaluated.column_keys != baseline.column_keys):
                raise ValueError("balance identities changed under an observation perturbation")
        target_jacobian[:, index] = (plus.target-minus.target)/(2*step)
        design_jacobian[:, :, index] = ((plus.matrix-minus.matrix)/(2*step)).T
    linearization = SharedObservationLinearization(
        target_jacobian, tuple(design_jacobian), tuple(coordinate_keys))
    return baseline, linearization
