"""Finite mechanism/partition search with one final score, including null.

The log-RSS penalty is heuristic under correlated weak errors. It is not a
confidence probability. This independent driver does not rescore old results.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .problem import Family, WeakView


@dataclass(frozen=True)
class Mechanism:
    name: str
    family: Family
    groups: tuple[int | None, ...]

    def __post_init__(self):
        labels, canonical = {}, []
        for group in self.groups:
            if group is None:
                canonical.append(None)
            else:
                if isinstance(group, (bool, np.bool_)) or not isinstance(group, (int, np.integer)) or group < 0:
                    raise ValueError("Invalid partition label")
                labels.setdefault(int(group), len(labels))
                canonical.append(labels[int(group)])
        if not self.name or not labels:
            raise ValueError("A mechanism needs a name and an active destination")
        object.__setattr__(self, "groups", tuple(canonical))

    @property
    def group_count(self):
        return len({g for g in self.groups if g is not None})


@dataclass(frozen=True)
class Candidate:
    mechanisms: tuple[Mechanism, ...] = ()

    def __post_init__(self):
        if len({m.name for m in self.mechanisms}) != len(self.mechanisms):
            raise ValueError("Duplicate mechanism identity in candidate")
        # Mechanism order determines coefficient layout, so freeze it canonically.
        object.__setattr__(self, "mechanisms", tuple(sorted(self.mechanisms, key=lambda m: m.name)))

    @property
    def key(self):
        return tuple((m.name, m.family.name, m.groups) for m in self.mechanisms)

    @property
    def flat_bounds(self):
        return tuple(b for m in self.mechanisms for _ in range(m.group_count) for b in m.family.bounds)

    @property
    def center(self):
        return np.asarray([v for m in self.mechanisms for _ in range(m.group_count) for v in m.family.center])

    def split(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(self.flat_bounds),) or not np.isfinite(values).all():
            raise ValueError("Internal parameter vector differs from candidate layout")
        result, offset = [], 0
        for m in self.mechanisms:
            width = len(m.family.bounds)
            result.append(tuple(values[offset+g*width:offset+(g+1)*width] for g in range(m.group_count)))
            offset += width*m.group_count
        return tuple(result)

    def parameter_count(self, bases):
        if any(len(m.groups) != len(bases) for m in self.mechanisms):
            raise ValueError("Candidate destination count differs from equations")
        return (sum(b.shape[1] for b in bases)
                + sum(sum(g is not None for g in m.groups) for m in self.mechanisms)
                + len(self.flat_bounds))


def partitions(equation_count):
    """Every nonempty support and canonical set partition of its destinations."""
    def restricted_growth(n, prefix=()):
        if len(prefix) == n:
            yield prefix
            return
        for label in range(max(prefix, default=-1)+2):
            yield from restricted_growth(n, prefix+(label,))
    if not isinstance(equation_count, int) or isinstance(equation_count, bool) or equation_count < 1:
        raise ValueError("Invalid equation count")
    for support in itertools.product((False, True), repeat=equation_count):
        if not any(support):
            continue
        for labels in restricted_growth(sum(support)):
            active = iter(labels)
            yield tuple(next(active) if present else None for present in support)


def enumerate_candidates(families_by_mechanism, equation_count):
    """Explicit candidate space; callers must declare its scientific scope."""
    options = []
    for name, families in sorted(families_by_mechanism.items()):
        options.append((None, *(Mechanism(name, family, groups) for family in families
                                 for groups in partitions(equation_count))))
    if not options:
        return (Candidate(),)
    return tuple(Candidate(tuple(m for m in choices if m is not None))
                 for choices in itertools.product(*options))


@dataclass(frozen=True)
class ScoreConvention:
    row_counts: tuple[int, ...]
    floors: tuple[float, ...]

    @classmethod
    def from_targets(cls, targets):
        if not targets or any(y.ndim != 1 or not y.size or not np.isfinite(y).all() for y in targets):
            raise ValueError("Invalid targets for common score")
        if sum(y.size for y in targets) <= 1:
            raise ValueError("Common score needs more than one row")
        return cls(tuple(y.size for y in targets),
                   tuple(np.finfo(np.float64).eps*max(float(y@y), 1.) for y in targets))

    def objective(self, rss):
        rss = np.asarray(rss, dtype=np.float64)
        if rss.shape != (len(self.row_counts),) or not np.isfinite(rss).all() or np.any(rss < 0):
            raise ValueError("Invalid equation residuals")
        return float(sum(n*np.log(max(r, floor)/n) for n, r, floor in
                         zip(self.row_counts, rss, self.floors)))

    def score(self, rss, parameter_count):
        if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count < 0:
            raise ValueError("Invalid nominal parameter count")
        return self.objective(rss) + parameter_count*np.log(sum(self.row_counts))


def design(view: WeakView, candidate: Candidate, parameters):
    bases = view.bases
    candidate.parameter_count(bases)
    by_mechanism = [tuple(view.atom(m.family, p) for p in block)
                    for m, block in zip(candidate.mechanisms, candidate.split(parameters))]
    matrices = []
    for e, base in enumerate(bases):
        columns = [base]
        for m, blocks in zip(candidate.mechanisms, by_mechanism):
            if m.groups[e] is not None:
                columns.append(blocks[m.groups[e]][e][:, None])
        matrices.append(np.column_stack(columns))
    if any(not np.isfinite(a).all() for a in matrices):
        raise ValueError("Nonfinite mechanism design")
    return tuple(matrices)


def conditional_profile(view, candidate, parameters):
    matrices = design(view, candidate, parameters)
    coefficients = tuple(np.linalg.lstsq(a, y, rcond=None)[0] for a, y in zip(matrices, view.targets))
    rss = tuple(float((y-a@c)@(y-a@c)) for a, y, c in zip(matrices, view.targets, coefficients))
    return coefficients, rss


@dataclass(frozen=True)
class FitBudget:
    seeds: tuple[int, ...] = (11011, 11012, 11013)
    initial_fractions: tuple[float, ...] = (.25, .5, .75)
    global_iterations: int = 18
    population_size: int = 10
    local_iterations: int = 3000
    maximum_objective_calls: int = 60000
    maximum_cpu_seconds: float = 600.

    def __post_init__(self):
        if (not self.seeds or len(self.seeds) != len(self.initial_fractions) or len(set(self.seeds)) != len(self.seeds)
                or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds)
                or any(not np.isfinite(f) or not 0 < f < 1 for f in self.initial_fractions)):
            raise ValueError("Invalid declared restarts")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
               (self.population_size, self.local_iterations, self.maximum_objective_calls)):
            raise ValueError("Invalid optimizer budget")
        if (isinstance(self.global_iterations, bool) or not isinstance(self.global_iterations, int)
                or self.global_iterations < 0 or not np.isfinite(self.maximum_cpu_seconds)
                or self.maximum_cpu_seconds <= 0):
            raise ValueError("Invalid global/time budget")


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class FittedCandidate:
    candidate: Candidate
    parameters: np.ndarray
    coefficients: tuple[np.ndarray, ...]
    equation_rss: tuple[float, ...]
    training_objective: float
    final_score: float
    nominal_parameter_count: int
    training_scale_sha256: str
    status: str
    objective_calls: int
    cpu_seconds: float
    wall_seconds: float
    restarts: tuple[dict, ...]


def fit_candidate(view, candidate, budget=FitBudget()):
    if tuple(c.name for c in view.conditions) != view.scale.condition_names:
        raise ValueError("Estimator view must be the frozen training conditions")
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    convention = ScoreConvention.from_targets(view.targets)
    count = candidate.parameter_count(view.bases)
    calls, best, records = 0, None, []

    def objective(parameters):
        nonlocal calls, best
        if calls >= budget.maximum_objective_calls or time.process_time()-started_cpu >= budget.maximum_cpu_seconds:
            raise BudgetExceeded("Declared per-program objective/CPU budget exhausted")
        calls += 1
        coefficients, rss = conditional_profile(view, candidate, parameters)
        value = convention.objective(rss)
        if best is None or value < best[0]:
            best = (value, np.asarray(parameters).copy(), coefficients, rss)
        return value

    if not candidate.mechanisms:
        objective(np.empty(0))
        status = "linear_null"
    else:
        bounds = np.asarray(candidate.flat_bounds)
        lower, span = bounds[:, 0], bounds[:, 1]-bounds[:, 0]
        status = "completed_unconverged"
        for seed, fraction in zip(budget.seeds, budget.initial_fractions):
            record = {"seed": seed, "initial_fraction": fraction, "global_success": None,
                      "local_success": None, "budget_exhausted": False}
            before = calls
            try:
                start = lower+fraction*span
                initial_value = objective(start)
                if budget.global_iterations:
                    global_fit = differential_evolution(
                        objective, candidate.flat_bounds, seed=seed, polish=False,
                        maxiter=budget.global_iterations, popsize=budget.population_size,
                        tol=1e-8, workers=1, updating="immediate")
                    record["global_success"] = bool(global_fit.success)
                    if global_fit.fun < initial_value:
                        start = global_fit.x
                # Unit-box coordinates are an exact parameter change, not a
                # precision claim; signed external amplitudes remain profiled.
                local_fit = minimize(lambda u: objective(lower+u*span), (start-lower)/span,
                                     method="Nelder-Mead", bounds=[(0., 1.)]*len(span),
                                     options={"maxiter": budget.local_iterations,
                                              "xatol": 1e-9, "fatol": 1e-10})
                record["local_success"] = bool(local_fit.success)
                record["terminal_training_objective"] = float(local_fit.fun)
            except BudgetExceeded:
                record["budget_exhausted"] = True
                status = "budget_exhausted"
            record["objective_calls"] = calls-before
            records.append(record)
            if record["budget_exhausted"]:
                break
        # Record actual convergence of the searches; precision is unassessed.
        if status != "budget_exhausted" and any(r["local_success"] or r["global_success"] for r in records):
            status = "completed_with_converged_search"
    if best is None:
        raise BudgetExceeded("No finite training estimate before budget exhaustion")
    value, parameters, coefficients, rss = best
    return FittedCandidate(candidate, parameters, coefficients, rss, value,
                           convention.score(rss, count), count, view.scale.fingerprint, status, calls,
                           time.process_time()-started_cpu, time.perf_counter()-started_wall,
                           tuple(records))


def predict_frozen(view, fit):
    """Apply trained parameters and amplitudes; never fit heldout coefficients."""
    if view.scale.fingerprint != fit.training_scale_sha256:
        raise ValueError("Prediction scale differs from frozen training scale")
    matrices = design(view, fit.candidate, fit.parameters)
    if len(matrices) != len(fit.coefficients) or any(a.shape[1] != c.size for a, c in zip(matrices, fit.coefficients)):
        raise ValueError("Frozen coefficient layout differs from prediction view")
    return tuple(a@c/w for a, c, w in zip(matrices, fit.coefficients, view.scale.target_weights))


def physical_coefficients(view, fit):
    if view.scale.fingerprint != fit.training_scale_sha256:
        raise ValueError("Coefficient units differ from frozen training scale")
    result = []
    for e, values in enumerate(fit.coefficients):
        norms = list(view.scale.base_norms[e])
        norms.extend(view.scale.norms_for(m.family)[e] for m in fit.candidate.mechanisms if m.groups[e] is not None)
        result.append(values/np.asarray(norms))
    return tuple(result)


def search(view, candidates, *, maximum_finalists=12, budget=FitBudget()):
    """Center surrogate then common-score refinement, with null always admitted."""
    if not isinstance(maximum_finalists, int) or isinstance(maximum_finalists, bool) or maximum_finalists < 1:
        raise ValueError("Invalid finalist quota")
    candidates = tuple(candidates)
    if not candidates or len({c.key for c in candidates}) != len(candidates):
        raise ValueError("Candidate space empty or duplicated")
    if Candidate() not in candidates:
        raise ValueError("Declared search must include null")
    if tuple(c.name for c in view.conditions) != view.scale.condition_names:
        raise ValueError("Search must use training data only")
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    convention = ScoreConvention.from_targets(view.targets)
    screened = []
    for index, candidate in enumerate(candidates):
        _, rss = conditional_profile(view, candidate, candidate.center)
        screened.append((convention.score(rss, candidate.parameter_count(view.bases)), index))
    screened.sort()
    null_index = candidates.index(Candidate())
    admitted = [i for _, i in screened if i != null_index][:maximum_finalists-1]+[null_index]
    fits = tuple(fit_candidate(view, candidates[i], budget) for i in admitted)
    ranked = tuple(sorted(fits, key=lambda f: (f.final_score, str(f.candidate.key))))
    return {"ranked": ranked, "screened_count": len(candidates), "admitted_indices": tuple(admitted),
            "screen_surrogate": "fixed_family_reference_common_score_not_optimized",
            "screen_may_omit_truth": True, "cpu_seconds": time.process_time()-started_cpu,
            "wall_seconds": time.perf_counter()-started_wall,
            "scientific_qualification_established": False}
