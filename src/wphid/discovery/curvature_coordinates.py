"""Local coordinate diagnostics; no truth, priors, or likelihood changes."""
import numpy as np


def finite_difference_hessian(value_gradient, point, step=1e-5):
    point = np.asarray(point, dtype=float)
    columns = []
    for index in range(len(point)):
        offset = np.zeros_like(point)
        offset[index] = step
        columns.append((value_gradient(point+offset)[1]-value_gradient(point-offset)[1])/(2*step))
    raw = np.column_stack(columns)
    return (raw+raw.T)/2, float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw), 1e-30))


def curvature_summary(hessian):
    hessian = np.asarray(hessian, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    diagonal = np.sqrt(np.maximum(np.abs(np.diag(hessian)), 1e-30))
    scaled = hessian/diagonal[:, None]/diagonal[None, :]
    scaled_values = np.linalg.eigvalsh(scaled)
    return {"eigenvalues": eigenvalues.tolist(), "positive_definite": bool(eigenvalues[0] > 0),
        "spectral_condition_if_positive": float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > 0 else None,
        "diagonal_scaled_eigenvalues": scaled_values.tolist(),
        "diagonal_scaled_condition_if_positive": float(scaled_values[-1]/scaled_values[0]) if scaled_values[0] > 0 else None,
        "weakest_scaled_coordinate_direction": eigenvectors[:, 0].tolist()}


def whitening_coordinates(hessian, relative_floor=1e-8):
    """Invert regularized local curvature as coordinates, not objective penalty."""
    values, vectors = np.linalg.eigh(hessian)
    ceiling = float(np.max(np.abs(values)))
    if not np.isfinite(ceiling) or ceiling <= 0 or not 0 < relative_floor < 1:
        raise ValueError("finite nonzero curvature and fractional floor required")
    metric_values = np.maximum(values, relative_floor*ceiling)
    return (vectors/np.sqrt(metric_values))@vectors.T
