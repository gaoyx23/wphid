"""Sparse full-covariance objective including log determinant; no optimizer.

This evaluates a first-order Gaussian quasi-likelihood conditional on a supplied
observation factor and evaluated observation derivatives. It does not qualify
the noise law, nonlinear mean bias, data-dependent Jacobian likelihood, physical
calibration or an exact maximum-likelihood estimator. No implicit jitter, blocks
or independent-row approximation are introduced.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from wphid.coupled_balance_v1.noise import CovarianceRankFailure


class LikelihoodBudgetFailure(ValueError):
    pass


@dataclass(frozen=True)
class QuasiLikelihoodResult:
    objective: float
    quadratic: float
    log_determinant: float
    linear_gradient: object
    covariance_nnz: int
    row_count: int


def _permutation_sign(permutation):
    seen = np.zeros(len(permutation), dtype=bool)
    cycles = 0
    for start in range(len(permutation)):
        if seen[start]:
            continue
        cycles += 1
        node = start
        while not seen[node]:
            seen[node] = True
            node = permutation[node]
    return -1 if (len(permutation)-cycles) % 2 else 1


def evaluate_quasi_likelihood(balance, linearization, coefficients, factor, *,
                              max_covariance_pairs=10_000_000, rank_rtol=1e-12,
                              compute_linear_gradient=False, gradient_max_rows=128):
    """Evaluate .5*(r^T Sigma^-1 r + logdet Sigma), omitting a fixed constant.

    Sigma(a)=(J_b-sum a_j J_Gj) F F^T (... )^T. For large systems the
    score uses sparse LU; analytic coefficient gradients are explicitly bounded
    to a small dense inverse. This is not a linear coefficient profile, candidate
    selector or production optimizer. Fixed row units cancel algebraically.
    """
    rows, columns = balance.matrix.shape
    if (linearization.target_jacobian.shape[0] != rows or
            len(linearization.column_jacobians) != columns):
        raise ValueError("balance and observation derivatives have incompatible shapes")
    if not np.isfinite(rank_rtol) or not 0 < rank_rtol < .1:
        raise ValueError("rank tolerance must be finite and between zero and .1")
    if not isinstance(max_covariance_pairs, int) or max_covariance_pairs < 1:
        raise ValueError("covariance pair budget must be an explicit positive integer")
    if compute_linear_gradient and rows > gradient_max_rows:
        raise LikelihoodBudgetFailure("exact coefficient gradient cap exceeded; no dense global inverse was formed")
    factor = csr_matrix(factor)
    residual = balance.residual(coefficients)
    noise = linearization.noise_operator(coefficients, factor)
    root = csr_matrix(noise.root)
    root.eliminate_zeros()
    counts = np.diff(root.tocsc().indptr).astype(np.int64)
    pair_bound = int(sum(int(count)**2 for count in counts))
    if pair_bound > max_covariance_pairs:
        raise LikelihoodBudgetFailure("covariance pair budget exceeded before matrix multiplication")
    variance = np.asarray(root.multiply(root).sum(axis=1)).ravel()
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise CovarianceRankFailure("zero or unresolved covariance diagonal; no jitter added")
    scale = np.sqrt(variance)
    normalized_root = root.multiply((1/scale)[:, None]).tocsr()
    covariance = (normalized_root @ normalized_root.T).tocsc()
    covariance.sum_duplicates()
    covariance.eliminate_zeros()
    if not np.all(np.isfinite(covariance.data)):
        raise CovarianceRankFailure("nonfinite covariance")
    try:
        lu = splu(covariance)
    except RuntimeError as error:
        raise CovarianceRankFailure("singular covariance; no jitter added") from error
    pivot = lu.U.diagonal()
    if np.any(np.abs(pivot) <= rank_rtol*max(float(np.max(np.abs(pivot))), 1.)):
        raise CovarianceRankFailure("numerically unresolved normalized covariance; no jitter added")
    determinant_sign = _permutation_sign(lu.perm_r)*_permutation_sign(lu.perm_c)*int(np.prod(np.sign(pivot)))
    if determinant_sign <= 0:
        raise CovarianceRankFailure("Gram covariance determinant unresolved; no repair added")
    normalized_residual = residual/scale
    normalized_solution = lu.solve(normalized_residual)
    solution = normalized_solution/scale
    quadratic = float(normalized_residual @ normalized_solution)
    log_determinant = float(np.sum(np.log(np.abs(pivot)))+2*np.sum(np.log(scale)))
    gradient = None
    if compute_linear_gradient:
        inverse = lu.solve(np.eye(rows))/scale[:, None]/scale[None, :]
        original_root = root
        gradient = np.empty(columns)
        for index, column in enumerate(linearization.column_jacobians):
            derivative_root = -column @ factor
            derivative_covariance = (derivative_root @ original_root.T+original_root @ derivative_root.T).toarray()
            gradient[index] = (-balance.matrix[:, index] @ solution+
                .5*np.einsum("ij,ji->", inverse, derivative_covariance)-
                .5*solution @ derivative_covariance @ solution)
    objective = .5*(quadratic+log_determinant)
    if (not np.isfinite(objective) or quadratic < 0 or
            gradient is not None and not np.all(np.isfinite(gradient))):
        raise CovarianceRankFailure("unresolved covariance solve or objective")
    return QuasiLikelihoodResult(objective, quadratic, log_determinant, gradient, covariance.nnz, rows)
