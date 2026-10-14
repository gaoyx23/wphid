"""Whole-product correction with independent node/field Gaussian covariance."""

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class VectorJet:
    value: np.ndarray
    jacobians: tuple
    diagonal_hessian_sums: tuple

    def __post_init__(self):
        values = np.asarray(self.value, dtype=float)
        jacobians = tuple(csr_matrix(j, dtype=float) for j in self.jacobians)
        curvatures = tuple(np.asarray(h, dtype=float) for h in self.diagonal_hessian_sums)
        if (values.ndim != 1 or not len(jacobians) or len(jacobians) != len(curvatures)
                or any(j.shape[0] != len(values) or j.shape != jacobians[0].shape for j in jacobians)
                or any(h.shape != values.shape for h in curvatures)
                or not all(np.isfinite(x).all() for x in (values, *curvatures, *(j.data for j in jacobians)))):
            raise ValueError('Invalid weak-value derivatives')
        object.__setattr__(self, 'value', values)
        object.__setattr__(self, 'jacobians', jacobians)
        object.__setattr__(self, 'diagonal_hessian_sums', curvatures)


def corrected_inner(left, right, variances):
    variances = np.asarray(variances, dtype=float)
    if (left.value.shape != right.value.shape or left.jacobians[0].shape != right.jacobians[0].shape
            or len(left.jacobians) != len(right.jacobians) or variances.shape != (len(left.jacobians),)
            or not np.isfinite(variances).all() or np.any(variances < 0)):
        raise ValueError('Invalid whole-product covariance layout')
    value = float(left.value @ right.value)
    for variance, lj, rj, lh, rh in zip(variances, left.jacobians, right.jacobians,
                                      left.diagonal_hessian_sums, right.diagonal_hessian_sums):
        value -= .5*variance*(float(lh @ right.value + left.value @ rh) + 2*float(lj.multiply(rj).sum()))
    return value


def family_derivatives(family, states, parameters):
    """Analytic first and diagonal second state derivatives of nine families."""
    t, o, d = (np.asarray(s, dtype=float) for s in states)
    parameters = np.asarray(parameters, dtype=float)
    if (parameters.shape != (len(family.bounds),) or not np.isfinite(parameters).all()
            or any(s.shape != t.shape or not np.isfinite(s).all() for s in (t, o, d))
            or np.any(t <= 0) or np.any(o <= 0) or np.any(d <= 0) or np.any(d >= 1)):
        raise ValueError('Family derivatives outside declared observation domain')
    value = family.values((t, o, d), parameters)
    first = [np.zeros_like(t) for _ in range(3)]
    second = [np.zeros_like(t) for _ in range(3)]
    name = family.name
    if name.startswith('arrhenius'):
        theta = parameters[0]
        first[0], second[0] = theta/t**2, -2*theta/t**3
        if name != 'arrhenius_damage':
            order = parameters[1]
            first[1], second[1] = order/o, -order/o**2
        if name != 'arrhenius_oxygen':
            order = parameters[2] if name == 'arrhenius_oxygen_damage' else parameters[1]
            first[2], second[2] = -order/(1-d), -order/(1-d)**2
    else:
        order = parameters[0]
        if name in ('oxygen_damage_power', 'oxygen_remaining_power'):
            first[1], second[1] = 1/o, -1/o**2
        if name == 'oxygen_power_damage':
            first[1], second[1] = order/o, -order/o**2
            first[2], second[2] = 1/d, -1/d**2
        elif name in ('damage_power', 'oxygen_damage_power'):
            first[2], second[2] = order/d, -order/d**2
        elif name in ('remaining_life_power', 'oxygen_remaining_power'):
            first[2], second[2] = -order/(1-d), -order/(1-d)**2
        elif name == 'damage_logistic_power':
            first[2], second[2] = 1/d-order/(1-d), -1/d**2-order/(1-d)**2
        else:
            raise ValueError('Family derivative not implemented')
    return value, tuple(value*g for g in first), tuple(value*(g*g+h) for g, h in zip(first, second))
