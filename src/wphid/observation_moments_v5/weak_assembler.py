"""Fixed common anchors for the declared three-field weak-PDE library.

The first application uses fixed identity moment weights. It does not claim
an optimal covariance, a likelihood, or compatibility with seven-field data.
"""

import hashlib
import json

import numpy as np
from scipy.sparse import csr_matrix

from wphid.final_candidate_v1.problem import paper_families
from .estimator import CommonCriterion, Model
from .jets import VectorJet, corrected_inner, family_derivatives


def fixed_anchors():
    families = paper_families()
    reaction = next(f for f in families if f.name == 'arrhenius_oxygen_damage')
    result = [(f.name + ':center', f, tuple(f.center)) for f in families]
    for axis in range(3):
        for fraction in (.25, .75):
            values = list(reaction.center)
            lo, hi = reaction.bounds[axis]
            values[axis] = lo + fraction*(hi-lo)
            result.append((f'arrhenius:axis{axis}:{fraction}', reaction, tuple(values)))
    return tuple(result)


class WeakMomentAssembler:
    def __init__(self, conditions, operators, sigma_by_condition, *, corrected=True,
                 field_units=(1000., 1., 1.)):
        self.conditions = tuple(conditions)
        self.operators = tuple(csr_matrix(o) for o in operators)
        self.units = np.asarray(field_units, dtype=float)
        sigma = tuple(np.asarray(s, dtype=float) for s in sigma_by_condition)
        if (not self.conditions or len(self.operators) != 4 or self.units.shape != (3,)
                or not np.isfinite(self.units).all() or np.any(self.units <= 0)
                or len(sigma) != len(self.conditions)
                or any(s.shape != (3,) or not np.isfinite(s).all() or np.any(s < 0) for s in sigma)):
            raise ValueError('Invalid fixed common weak-moment layout')
        self.variances = tuple(s*s if corrected else np.zeros(3) for s in sigma)
        self.corrected = bool(corrected)
        self.anchor_spec = fixed_anchors()
        self.anchor_names = ('constant', *(f'base:{e}:{k}' for e in range(3) for k in range(3)),
                             *(name for name, _, _ in self.anchor_spec))
        self.blocks, target, rows = [], [], []
        projection = self.operators[2]
        self.zero = csr_matrix(projection.shape)
        self.zeros = np.zeros(projection.shape[0])
        for condition, variance in zip(self.conditions, self.variances):
            difference = projection-condition.projection
            if difference.nnz and np.max(np.abs(difference.data)) > 2e-13:
                raise ValueError('Reaction weak operator differs from saved observation')
            bases = []
            for e, state in enumerate(condition.states):
                if not np.allclose(np.column_stack([o @ state for o in self.operators[:3]]), condition.bases[e], rtol=2e-11, atol=2e-12):
                    raise ValueError('Base operator differs from saved observation')
                bases.append(tuple(self.linear_jet(condition.bases[e][:, k]/self.units[e], o/self.units[e], e)
                                   for k, o in enumerate(self.operators[:3])))
            anchors = [VectorJet(np.ones(projection.shape[0]), (self.zero,)*3, (self.zeros,)*3)]
            anchors.extend(j for values in bases for j in values)
            for _, family, parameters in self.anchor_spec:
                reference = float(family.values((np.array([1000.]), np.array([1.]), np.array([.5])), parameters)[0])
                if reference <= 0 or not np.isfinite(reference):
                    raise ValueError('Invalid deterministic anchor unit')
                anchors.append(self.reaction_jet(condition.states, family, parameters, 1/reference))
            h = np.stack([a.value for a in anchors])
            bias = np.stack([.5*sum(v*c for v, c in zip(variance, a.diagonal_hessian_sums)) for a in anchors])
            # Precontract all sparse jets with the common projection; each
            # nonlinear evaluation now uses small dense matrix/vector products.
            bare = np.asarray(projection.T @ h.T).T
            alpha = np.asarray(projection.T @ (h-bias).T).T
            cross = tuple(np.stack([np.asarray(a.jacobians[s].multiply(projection).sum(axis=0)).ravel()
                                    for a in anchors])*variance[s] for s in range(3))
            base_maps = []
            for e in range(3):
                target_jet = self.linear_jet(condition.targets[e]/self.units[e], self.operators[3]/self.units[e], e)
                target.extend(corrected_inner(a, target_jet, variance) for a in anchors)
                base_maps.append(np.asarray([[corrected_inner(a, b, variance) for b in bases[e]] for a in anchors]))
                rows.extend(f'{condition.name}/equation{e}/{name}' for name in self.anchor_names)
            self.blocks.append({'states': condition.states, 'variance': variance,
                                'bare': bare, 'alpha': alpha, 'cross': cross, 'bases': tuple(base_maps)})
        self.target = np.asarray(target)
        self.row_ids = tuple(rows)
        declaration = {'anchors': [(name, family.name, parameters) for name, family, parameters in self.anchor_spec],
                       'field_units': self.units.tolist(), 'conditions': [c.name for c in self.conditions],
                       'corrected': self.corrected, 'rows': self.row_ids}
        self.declaration_sha256 = hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest()

    def linear_jet(self, values, operator, field):
        jacobians = [self.zero]*3
        jacobians[field] = operator
        return VectorJet(values, tuple(jacobians), (self.zeros,)*3)

    def reaction_jet(self, states, family, parameters, factor):
        value, first, second = family_derivatives(family, states, parameters)
        projection = self.operators[2]
        return VectorJet((projection @ value)*factor,
                         tuple(projection.multiply(g)*factor for g in first),
                         tuple((projection @ h)*factor for h in second))

    def response(self, block, family, parameters):
        value, first, second = family_derivatives(family, block['states'], parameters)
        result = block['alpha'] @ value
        for variance, gradient, curvature, cross in zip(block['variance'], first, second, block['cross']):
            result -= .5*variance*(block['bare'] @ curvature) + cross @ gradient
        return result

    def coefficient_layout(self, candidate):
        counts = tuple(3 + sum(m.groups[e] is not None for m in candidate.mechanisms) for e in range(3))
        return counts, np.r_[0, np.cumsum(counts)]

    def matrix(self, candidate, parameters):
        counts, offsets = self.coefficient_layout(candidate)
        parameters = candidate.split(parameters)
        width = len(self.anchor_names)
        matrix = np.zeros((len(self.row_ids), int(offsets[-1])))
        for index, block in enumerate(self.blocks):
            reactions = { (m.name, g): self.response(block, m.family, values)
                          for m, groups in zip(candidate.mechanisms, parameters) for g, values in enumerate(groups) }
            for e in range(3):
                rows = slice((index*3+e)*width, (index*3+e+1)*width)
                begin = int(offsets[e])
                matrix[rows, begin:begin+3] = block['bases'][e]
                column = begin+3
                for mechanism in candidate.mechanisms:
                    if mechanism.groups[e] is not None:
                        matrix[rows, column] = reactions[mechanism.name, mechanism.groups[e]]/self.units[e]
                        column += 1
        return matrix

    def model(self, candidate):
        counts, _ = self.coefficient_layout(candidate)
        return Model(repr(candidate.key), candidate.flat_bounds, sum(counts),
                     lambda parameters: self.matrix(candidate, parameters))

    def criterion(self):
        # Identity is a fixed metric, not an independence claim about moments.
        return CommonCriterion(self.target, self.row_ids)

    def physical_coefficients(self, candidate, values):
        _, offsets = self.coefficient_layout(candidate)
        values = np.asarray(values)
        if values.shape != (int(offsets[-1]),):
            raise ValueError('Physical coefficient layout differs from candidate')
        return tuple(values[int(offsets[e]):int(offsets[e+1])] for e in range(3))
