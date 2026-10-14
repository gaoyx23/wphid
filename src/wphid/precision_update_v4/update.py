"""Apply one prospectively declared step, never optimize or truth-select it."""

import numpy as np


def one_step(risk, parameters, bounds, predicted_unit_step, maximum_unit_step=.1):
    p, bounds, step = (np.asarray(x, dtype=np.float64) for x in (parameters, bounds, predicted_unit_step))
    if (p.ndim != 1 or bounds.shape != (len(p), 2) or step.shape != p.shape
            or not all(np.isfinite(x).all() for x in (p, bounds, step))
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError('Invalid fixed step layout')
    if np.max(np.abs(step)) > maximum_unit_step:
        raise ValueError('Fixed local step exceeds the declared radius; no damping')
    proposed = p + (bounds[:, 1]-bounds[:, 0])*step
    if np.any(proposed <= bounds[:, 0]) or np.any(proposed >= bounds[:, 1]):
        raise ValueError('Fixed update outside physical box; no clipping or fallback')
    # Exactly one fixed-parameter profile (three linear solves), no iteration.
    result = risk.profile(proposed, mean_correction=True, gram_correction=True)
    if not np.isfinite(result.objective) or any(not np.isfinite(c).all() for c in result.coefficients):
        raise ValueError('Nonfinite corrected conditional profile')
    return proposed, result


def physical_units(scale, family, coefficients):
    if len(coefficients) != 3:
        raise ValueError('Expected three equation coefficients')
    result = []
    for e, values in enumerate(coefficients):
        values = np.asarray(values, dtype=float)
        norms = np.r_[scale.base_norms[e], scale.norms_for(family)[e]]
        if values.shape != norms.shape or np.any(norms <= 0):
            raise ValueError('Coefficient scale mismatch')
        result.append(values/norms)
    return tuple(result)
