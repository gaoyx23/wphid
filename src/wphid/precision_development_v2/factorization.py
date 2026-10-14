"""Factor the V1 equation-wise profile by candidate parameter connectivity.

No observations, candidate choices, parameter bounds or linear rank rules are
changed. This applies only to the V1 independent-linear-coefficient log-RSS
objective; cross-equation covariance or shared linear coefficients require a
different derivation. It is not an identifiability or recovery guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wphid.final_candidate_v1.solver import Candidate, ScoreConvention


@dataclass(frozen=True)
class ParameterBlock:
    mechanism_index: int
    group: int
    parameter_indices: tuple[int, ...]


@dataclass(frozen=True)
class Component:
    blocks: tuple[ParameterBlock, ...]
    equations: tuple[int, ...]
    parameter_indices: tuple[int, ...]
    candidate_key: tuple


@dataclass(frozen=True)
class Factorization:
    candidate_key: tuple
    equation_count: int
    components: tuple[Component, ...]
    inactive_equations: tuple[int, ...]


def factorize(candidate: Candidate, equation_count: int) -> Factorization:
    """Connect distinct mechanism/group nodes whenever they enter one equation."""
    if isinstance(equation_count, bool) or not isinstance(equation_count, int) or equation_count < 1:
        raise ValueError("Positive equation count required")
    if any(len(m.groups) != equation_count for m in candidate.mechanisms):
        raise ValueError("Candidate destination count differs from equations")
    nodes, node_for, offset = [], {}, 0
    for i, mechanism in enumerate(candidate.mechanisms):
        width = len(mechanism.family.bounds)
        for group in range(mechanism.group_count):
            node_for[i, group] = len(nodes)
            nodes.append(ParameterBlock(i, group, tuple(range(offset, offset+width))))
            offset += width
    parent = list(range(len(nodes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    active_nodes = []
    for e in range(equation_count):
        selected = tuple(node_for[i, m.groups[e]] for i, m in enumerate(candidate.mechanisms)
                         if m.groups[e] is not None)
        active_nodes.append(selected)
        for i in selected[1:]:
            a, b = root(selected[0]), root(i)
            parent[max(a, b)] = min(a, b)
    connected = {}
    for i in range(len(nodes)):
        connected.setdefault(root(i), []).append(i)
    components = []
    for key, indices in sorted(connected.items()):
        blocks = tuple(nodes[i] for i in indices)
        equations = tuple(e for e, selected in enumerate(active_nodes)
                          if selected and root(selected[0]) == key)
        components.append(Component(blocks, equations,
                                    tuple(j for block in blocks for j in block.parameter_indices), candidate.key))
    return Factorization(candidate.key, equation_count, tuple(components),
                         tuple(e for e, selected in enumerate(active_nodes) if not selected))


@dataclass(frozen=True)
class PreparedProfile:
    view: object
    candidate: Candidate
    factorization: Factorization
    bases: tuple[np.ndarray, ...]
    targets: tuple[np.ndarray, ...]
    convention: ScoreConvention
    nominal_parameter_count: int


def prepare_profile(view, candidate: Candidate) -> PreparedProfile:
    if tuple(c.name for c in view.conditions) != view.scale.condition_names:
        raise ValueError("Profile must use the frozen training conditions")
    bases, targets = view.bases, view.targets
    return PreparedProfile(view, candidate, factorize(candidate, len(bases)), bases, targets,
                           ScoreConvention.from_targets(targets), candidate.parameter_count(bases))


def component_profile(problem: PreparedProfile, component: Component, values):
    """Same V1 matrix/column order and lstsq rank convention on this component."""
    if component not in problem.factorization.components:
        raise ValueError("Component belongs to a different candidate")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(component.parameter_indices),) or not np.isfinite(values).all():
        raise ValueError("Component parameter vector differs from layout")
    atoms, offset = {}, 0
    for block in component.blocks:
        width = len(block.parameter_indices)
        family = problem.candidate.mechanisms[block.mechanism_index].family
        atoms[block.mechanism_index, block.group] = problem.view.atom(family, values[offset:offset+width])
        offset += width
    coefficients, rss = [], []
    for e in component.equations:
        columns = [problem.bases[e]]
        for i, mechanism in enumerate(problem.candidate.mechanisms):
            if mechanism.groups[e] is not None:
                columns.append(atoms[i, mechanism.groups[e]][e][:, None])
        matrix = np.column_stack(columns)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite mechanism design")
        coef = np.linalg.lstsq(matrix, problem.targets[e], rcond=None)[0]
        residual = problem.targets[e]-matrix@coef
        coefficients.append(coef)
        rss.append(float(residual@residual))
    objective = float(sum(problem.convention.row_counts[e]*np.log(
        max(r, problem.convention.floors[e])/problem.convention.row_counts[e])
        for e, r in zip(component.equations, rss)))
    return tuple(coefficients), tuple(rss), objective


@dataclass(frozen=True)
class ProfileAtPoint:
    parameters: np.ndarray
    coefficients: tuple[np.ndarray, ...]
    equation_rss: tuple[float, ...]
    component_objectives: tuple[float, ...]
    inactive_objective: float
    training_objective: float
    final_score: float


def profile_components(problem: PreparedProfile, parameters) -> ProfileAtPoint:
    """Evaluate a supplied point; no optimizer, parameter truth or selection."""
    problem.candidate.split(parameters)
    parameters = np.asarray(parameters, dtype=np.float64)
    coefficients, rss = [None]*len(problem.bases), [None]*len(problem.bases)
    objectives = []
    for component in problem.factorization.components:
        coefs, errors, objective = component_profile(problem, component, parameters[list(component.parameter_indices)])
        objectives.append(objective)
        for e, coef, error in zip(component.equations, coefs, errors):
            coefficients[e], rss[e] = coef, error
    for e in problem.factorization.inactive_equations:
        coef = np.linalg.lstsq(problem.bases[e], problem.targets[e], rcond=None)[0]
        residual = problem.targets[e]-problem.bases[e]@coef
        coefficients[e], rss[e] = coef, float(residual@residual)
    inactive = float(sum(problem.convention.row_counts[e]*np.log(
        max(rss[e], problem.convention.floors[e])/problem.convention.row_counts[e])
        for e in problem.factorization.inactive_equations))
    return ProfileAtPoint(parameters.copy(), tuple(coefficients), tuple(rss), tuple(objectives),
                          inactive, problem.convention.objective(rss),
                          problem.convention.score(rss, problem.nominal_parameter_count))
