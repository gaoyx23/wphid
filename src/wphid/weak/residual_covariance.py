"""First-order covariance for one weak PDE residual, including noisy atoms."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cholesky


def linearized_residual_covariance(operators, base_coefficients, amplitude,
                                  rate_gradients, noise_stds, equation):
    """Propagate independent nodal state noise, not just target noise.

    The residual is -time(u)-kz*axial(u)-kr*radial(u)-beta*mass(u)
    -amplitude*mass(rate)-mass(forcing). Forcing is assumed noise-free here.
    Correlations between overlapping weak windows are retained. This returns
    a per-equation marginal, NOT covariance between different equations.
    """
    gradients = [np.asarray(g, dtype=float).ravel() for g in rate_gradients]
    stds = np.asarray(noise_stds, dtype=float)
    base = np.asarray(base_coefficients, dtype=float)
    if (base.shape != (3,) or len(gradients) != len(stds)
            or not 0 <= equation < len(stds) or np.any(stds < 0)
            or not np.all(np.isfinite(stds)) or not np.all(np.isfinite(base))
            or not np.isfinite(amplitude)):
        raise ValueError("finite three base coefficients and aligned nonnegative state noise are required")
    mass = operators["mass"]
    covariance = np.zeros((mass.shape[0], mass.shape[0]))
    for field, (gradient, std) in enumerate(zip(gradients, stds)):
        if gradient.size != mass.shape[1] or not np.all(np.isfinite(gradient)):
            raise ValueError("finite rate gradient must match the nodal grid")
        jacobian = (-float(amplitude)*mass.multiply(gradient[None, :])).tocsr()
        if field == equation:
            jacobian = (jacobian - operators["time"] - base[0]*operators["axial"]
                        - base[1]*operators["radial"] - base[2]*mass).tocsr()
        covariance += std**2 * (jacobian @ jacobian.T).toarray()
    return (covariance + covariance.T)/2


def covariance_cholesky(covariance, *, relative_jitter=1e-10):
    """Fixed tiny numerical regularizer; no eigenvalue/threshold scan."""
    matrix = np.asarray(covariance, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not matrix.size or not np.all(np.isfinite(matrix))
            or not 0 < relative_jitter < 1e-3):
        raise ValueError("finite square covariance and small positive jitter required")
    scale = float(np.trace(matrix)/len(matrix))
    if scale <= 0:
        raise ValueError("calibrated covariance must have positive trace")
    jitter = relative_jitter * scale
    factor = cholesky((matrix+matrix.T)/2 + jitter*np.eye(len(matrix)), lower=True)
    return factor, {"relative_jitter": relative_jitter, "absolute_jitter": jitter,
                    "condition_number": float(np.linalg.cond(matrix)),
                    "cross_equation_covariance_used": False}
