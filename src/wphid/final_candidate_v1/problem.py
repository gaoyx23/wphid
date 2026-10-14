"""Truth-free weak observations and train-only reversible normalization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Family:
    name: str
    bounds: tuple[tuple[float, float], ...]
    center: tuple[float, ...]

    def __post_init__(self):
        widths = {"damage_power": 1, "damage_logistic_power": 1,
                  "remaining_life_power": 1, "oxygen_damage_power": 1,
                  "oxygen_remaining_power": 1, "oxygen_power_damage": 1,
                  "arrhenius_oxygen_damage": 3, "arrhenius_oxygen": 2,
                  "arrhenius_damage": 2}
        if self.name not in widths or len(self.bounds) != widths[self.name]:
            raise ValueError("Unknown family or incorrect internal width")
        if len(self.center) != len(self.bounds) or any(
                not np.isfinite([lo, hi, c]).all() or not lo < hi or not lo <= c <= hi
                for (lo, hi), c in zip(self.bounds, self.center)):
            raise ValueError("Family requires finite bounds and an interior reference")

    def values(self, states, parameters, *, xp=np):
        """Shared NumPy/PyTorch expression; oxygen reference is one declared unit."""
        t, o, d = states
        p = parameters
        if self.name == "damage_power":
            return d ** p[0]
        if self.name == "damage_logistic_power":
            return d * (1 - d) ** p[0]
        if self.name == "remaining_life_power":
            return (1 - d) ** p[0]
        if self.name == "oxygen_damage_power":
            return o * d ** p[0]
        if self.name == "oxygen_remaining_power":
            return o * (1 - d) ** p[0]
        if self.name == "oxygen_power_damage":
            return o ** p[0] * d
        rate = xp.exp(-p[0] / t)
        if self.name == "arrhenius_oxygen_damage":
            return rate * o ** p[1] * (1 - d) ** p[2]
        if self.name == "arrhenius_oxygen":
            return rate * o ** p[1]
        return rate * (1 - d) ** p[1]


def paper_families():
    """Declared finite library, not a learned or universal family inventory."""
    return (
        *(Family(n, ((0.3, 2.5),), (1.2,)) for n in
          ("damage_power", "damage_logistic_power", "remaining_life_power")),
        *(Family(n, ((0.2, 2.0),), (1.1,)) for n in
          ("oxygen_damage_power", "oxygen_remaining_power", "oxygen_power_damage")),
        Family("arrhenius_oxygen_damage", ((2000., 14000.), (0., 2.), (0., 3.)),
               (8000., 1., 1.5)),
        Family("arrhenius_oxygen", ((2000., 14000.), (0., 2.)), (8000., 1.)),
        Family("arrhenius_damage", ((2000., 14000.), (0., 3.)), (8000., 1.5)),
    )


def _frozen_array(value, ndim, label):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"Invalid {label}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WeakCondition:
    name: str
    bases: tuple[np.ndarray, ...]
    targets: tuple[np.ndarray, ...]
    states: tuple[np.ndarray, np.ndarray, np.ndarray]
    projection: csr_matrix

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Condition name must be explicit")
        bases = tuple(_frozen_array(b, 2, "base") for b in self.bases)
        targets = tuple(_frozen_array(b, 1, "target") for b in self.targets)
        states = tuple(_frozen_array(s, 1, "state") for s in self.states)
        projection = csr_matrix(self.projection, dtype=np.float64, copy=True)
        projection.sum_duplicates()
        projection.sort_indices()
        if not bases or len(bases) != len(targets) or len(states) != 3:
            raise ValueError("Equation or state counts differ")
        if any(b.shape[0] != y.size or y.size == 0 for b, y in zip(bases, targets)):
            raise ValueError("Empty or mismatched weak rows")
        if any(s.size != projection.shape[1] for s in states) or any(
                y.size != projection.shape[0] for y in targets):
            raise ValueError("State/projector dimensions differ")
        if not np.isfinite(projection.data).all() or projection.shape[1] == 0:
            raise ValueError("Invalid weak projection")
        t, o, d = states
        # Strict positivity avoids undefined exponent derivatives at zero.
        # The adapter never clips or silently changes supplied observations.
        if np.any(t <= 0) or np.any(o <= 0) or np.any(d <= 0) or np.any(d >= 1):
            raise ValueError("Require T>0, O>0 and 0<D<1 for differentiable powers")
        for values in (projection.data, projection.indices, projection.indptr):
            values.setflags(write=False)
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "projection", projection)

    def atom(self, family: Family, parameters):
        return np.asarray(self.projection @ family.values(self.states, parameters))


def validate_split(known, train, heldout):
    for names in (known, train, heldout):
        if len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("Invalid or duplicate condition names")
    if not train or not heldout or set(train) & set(heldout):
        raise ValueError("Training and heldout must be nonempty and disjoint")
    if set(train) | set(heldout) != set(known):
        raise ValueError("Unknown or uncovered condition")


@dataclass(frozen=True)
class TrainingScale:
    condition_names: tuple[str, ...]
    target_weights: tuple[float, ...]
    base_norms: tuple[np.ndarray, ...]
    atom_norms: tuple[tuple[str, tuple[float, ...]], ...]

    def norms_for(self, family):
        return dict(self.atom_norms)[family.name]

    @property
    def fingerprint(self):
        data = {"conditions": self.condition_names, "target_weights": self.target_weights,
                "base_norms": [n.tolist() for n in self.base_norms], "atom_norms": self.atom_norms}
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def fit(cls, conditions: Sequence[WeakCondition], families: Sequence[Family]):
        if not conditions or len({c.name for c in conditions}) != len(conditions):
            raise ValueError("Nonempty unique training conditions required")
        if not families or len({f.name for f in families}) != len(families):
            raise ValueError("Nonempty unique families required")
        equation_count = len(conditions[0].bases)
        widths = tuple(b.shape[1] for b in conditions[0].bases)
        if any(len(c.bases) != equation_count or
               tuple(b.shape[1] for b in c.bases) != widths for c in conditions):
            raise ValueError("Equation/base widths vary across conditions")
        weights, base_norms = [], []
        for e in range(equation_count):
            y = np.concatenate([c.targets[e] for c in conditions])
            weight = 1. / max(float(np.sqrt(np.mean(y*y))), 1e-12)
            b = np.vstack([c.bases[e] for c in conditions]) * weight
            norms = np.sqrt(np.mean(b*b, axis=0))
            if np.any(norms <= 1e-15):
                raise ValueError("A mandatory base column has zero training norm")
            norms.setflags(write=False)
            weights.append(weight)
            base_norms.append(norms)
        atom_norms = []
        for family in families:
            atom = np.concatenate([c.atom(family, family.center) for c in conditions])
            rms = float(np.sqrt(np.mean(atom*atom)))
            if not np.isfinite(rms) or rms <= 1e-15:
                raise ValueError("A reference atom has zero training norm")
            atom_norms.append((family.name, tuple(rms*w for w in weights)))
        return cls(tuple(c.name for c in conditions), tuple(weights), tuple(base_norms),
                   tuple(atom_norms))

    def apply(self, conditions: Sequence[WeakCondition]):
        return WeakView(tuple(conditions), self)


@dataclass(frozen=True)
class WeakView:
    conditions: tuple[WeakCondition, ...]
    scale: TrainingScale

    def __post_init__(self):
        if not self.conditions or len({c.name for c in self.conditions}) != len(self.conditions):
            raise ValueError("Nonempty unique view conditions required")
        for c in self.conditions:
            if len(c.bases) != len(self.scale.base_norms) or any(
                    b.shape[1] != n.size for b, n in zip(c.bases, self.scale.base_norms)):
                raise ValueError("View incompatible with frozen training scale")

    @property
    def bases(self):
        return tuple(np.vstack([c.bases[e] for c in self.conditions])*w/n
                     for e, (w, n) in enumerate(zip(self.scale.target_weights,
                                                    self.scale.base_norms)))

    @property
    def targets(self):
        return tuple(np.concatenate([c.targets[e] for c in self.conditions])*w
                     for e, w in enumerate(self.scale.target_weights))

    def atom(self, family: Family, parameters):
        raw = np.concatenate([c.atom(family, parameters) for c in self.conditions])
        norms = self.scale.norms_for(family)
        return tuple(raw*w/n for w, n in zip(self.scale.target_weights, norms))

    def builder(self, family):
        return partial(self.atom, family)

    def torch_builder(self, family):
        """Differentiable version of the same fixed linear weak operator."""
        import torch
        cached = []
        for c in self.conditions:
            coo = c.projection.tocoo()
            index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.int64)
            projector = torch.sparse_coo_tensor(index, torch.tensor(coo.data.copy()),
                                                 coo.shape, dtype=torch.float64,
                                                 check_invariants=True).coalesce()
            states = tuple(torch.tensor(s.copy(), dtype=torch.float64) for s in c.states)
            cached.append((states, projector))
        factors = tuple(w/n for w, n in zip(self.scale.target_weights,
                                            self.scale.norms_for(family)))

        def build(parameters):
            raw = torch.cat([torch.sparse.mm(p, family.values(s, parameters, xp=torch)[:, None])[:, 0]
                             for s, p in cached])
            return tuple(raw*f for f in factors)
        return build
