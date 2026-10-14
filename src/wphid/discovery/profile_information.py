"""Local information of nonlinear shapes after profiling linear coefficients.

Diagnostics only: overlapping weak windows and noisy regressors invalidate a
naive Gaussian confidence-interval interpretation of the singular values.
"""
from __future__ import annotations

import numpy as np


def conditional_profile_objective(bases, targets, atoms):
    """The existing per-equation log-RSS criterion for one active mechanism."""
    if not (len(bases) == len(targets) == len(atoms)) or not bases:
        raise ValueError("nonempty bases, targets, atoms must align")
    objective = 0.0
    for base, target, atom in zip(bases, targets, atoms):
        matrix = np.column_stack((base, atom))
        coefficient = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - matrix @ coefficient
        floor = np.finfo(float).eps * max(float(target @ target), 1.0)
        objective += len(target) * np.log(max(float(residual @ residual), floor)/len(target))
    return float(objective)


def diagnose_profile_information(bases, targets, builder, parameters, groups, *, step_fraction=1e-4):
    """Residualize shape derivatives against base AND amplitude columns.

    Derivatives are in relative-parameter coordinates. Report separate groups;
    this primitive assumes a single mechanism, active in every supplied equation.
    Estimated residual scales weight equations, but do not calibrate uncertainty.
    """
    if not 0 < step_fraction < 0.1:
        raise ValueError("step_fraction must be small and positive")
    if len(groups) != len(bases) or any(g is None for g in groups):
        raise ValueError("one active parameter group per equation is required")
    values = [np.asarray(p, dtype=float) for p in parameters]
    if any(v.ndim != 1 or not np.all(np.isfinite(v)) or np.any(v == 0) for v in values):
        raise ValueError("relative-coordinate diagnostics require finite nonzero parameter vectors")
    atom = [builder(values[group])[equation] for equation, group in enumerate(groups)]
    result = []
    for group, parameter in enumerate(values):
        equations = [e for e, g in enumerate(groups) if g == group]
        if not equations:
            raise ValueError("parameter groups must not contain unused labels")
        derivatives = []
        for coordinate in range(len(parameter)):
            delta = step_fraction * abs(parameter[coordinate])
            left, right = parameter.copy(), parameter.copy()
            left[coordinate] -= delta
            right[coordinate] += delta
            aa, bb = builder(left), builder(right)
            derivatives.append([(bb[e]-aa[e])/(2*delta)*parameter[coordinate] for e in equations])
        blocks = []
        for local_index, equation in enumerate(equations):
            base, response = bases[equation], targets[equation]
            matrix = np.column_stack((base, atom[equation]))
            coefficient = np.linalg.lstsq(matrix, response, rcond=None)[0]
            jacobian = np.column_stack([column[local_index] for column in derivatives]) * coefficient[-1]
            # Normalize columns only to stabilize the nuisance projection.
            norms = np.linalg.norm(matrix, axis=0)
            active = norms > 0
            nuisance = matrix[:, active]/norms[active]
            residualized = jacobian - nuisance @ np.linalg.lstsq(nuisance, jacobian, rcond=None)[0]
            residual = response - matrix @ coefficient
            sigma = max(float(np.sqrt(np.mean(residual**2))), np.sqrt(np.finfo(float).eps))
            blocks.append(residualized/sigma)
        combined = np.vstack(blocks)
        _, singular, right = np.linalg.svd(combined, full_matrices=False)
        ratio = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.0
        result.append({"group": group, "equations": equations,
            "relative_parameter_singular_values": singular.tolist(),
            "smallest_to_largest_singular_ratio": ratio,
            "weakest_relative_parameter_direction": right[-1].tolist(),
            "calibrated_parameter_uncertainty": False})
    return result
