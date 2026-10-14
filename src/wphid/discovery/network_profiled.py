"""Joint profiling of nonlinear mechanisms across state equations.

Parameter sharing is optional.  The legacy helpers model one parameter vector
per mechanism; :func:`profiled_partitioned_mechanism_fit` additionally lets
the data choose which destination equations share a vector.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize


NetworkAtomBuilder = Callable[[np.ndarray], Sequence[np.ndarray]]


@dataclass(frozen=True)
class NetworkProfiledFit:
    """A shared mechanism with equation-specific external coefficients."""

    parameters: np.ndarray
    coefficients: tuple[np.ndarray, ...]
    equation_rss: np.ndarray
    bic: float
    active_equations: tuple[bool, ...]
    success: bool


@dataclass(frozen=True)
class MultiMechanismProfiledFit:
    """Multiple shared mechanisms with mechanism-specific directed topologies."""

    parameters: tuple[np.ndarray, ...]
    coefficients: tuple[np.ndarray, ...]
    equation_rss: np.ndarray
    bic: float
    active_equations: tuple[tuple[bool, ...], ...]
    success: bool


@dataclass(frozen=True)
class PartitionedMechanismProfiledFit:
    """Mechanisms with equation topologies and optional parameter sharing.

    ``parameter_groups[k][e]`` is ``None`` when mechanism ``k`` does not enter
    equation ``e``.  Equal integer labels identify destination equations that
    share one inner-parameter vector.  Labels are canonicalized by first
    appearance, so ``(4, 4, 9)`` is reported as ``(0, 0, 1)``.
    """

    parameters: tuple[tuple[np.ndarray, ...], ...]
    coefficients: tuple[np.ndarray, ...]
    equation_rss: np.ndarray
    bic: float
    parameter_groups: tuple[tuple[int | None, ...], ...]
    success: bool


def _validate_problem(base_matrices, targets, active_equations):
    bases = tuple(np.asarray(matrix, dtype=float) for matrix in base_matrices)
    responses = tuple(np.asarray(target, dtype=float) for target in targets)
    active = tuple(bool(value) for value in active_equations)
    if not bases or len(bases) != len(responses) or len(bases) != len(active):
        raise ValueError("bases, targets, and active_equations must have equal nonzero length")
    for base, response in zip(bases, responses):
        if (base.ndim != 2 or response.ndim != 1
                or base.shape[0] != response.size
                or not np.all(np.isfinite(base))
                or not np.all(np.isfinite(response))):
            raise ValueError("each finite base matrix must match its one-dimensional target")
    return bases, responses, active


def _conditional_fit(bases, responses, atoms, active):
    coefficients, rss = [], []
    for base, response, atom, include in zip(bases, responses, atoms, active):
        vector = np.asarray(atom, dtype=float)
        if vector.shape != response.shape or not np.all(np.isfinite(vector)):
            raise ValueError("each mechanism atom must match its equation target")
        matrix = np.column_stack((base, vector)) if include else base
        estimate = np.linalg.lstsq(matrix, response, rcond=None)[0]
        residual = response-matrix@estimate
        coefficients.append(estimate)
        rss.append(max(float(residual@residual), np.finfo(float).tiny))
    return tuple(coefficients), np.asarray(rss)


def profiled_shared_mechanism_fit(
    base_matrices: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    atom_builder: NetworkAtomBuilder,
    active_equations: Sequence[bool],
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int = 0,
    global_iterations: int = 60,
) -> NetworkProfiledFit:
    """Fit shared inner parameters and equation-specific outer coefficients.

    The objective is the summed per-equation Gaussian profile likelihood, so a
    high-magnitude field cannot dominate merely through its physical units.
    Candidate topology is explicit in ``active_equations`` and is included in
    the BIC through each admitted equation-specific amplitude.
    """
    bases, responses, active = _validate_problem(
        base_matrices, targets, active_equations)
    if not any(active):
        raise ValueError("a shared mechanism must enter at least one equation")
    if not bounds or any(low >= high for low, high in bounds):
        raise ValueError("each nonlinear parameter bound must satisfy low < high")

    def evaluate(parameters):
        atoms = tuple(atom_builder(np.asarray(parameters, dtype=float)))
        if len(atoms) != len(responses):
            return None
        try:
            return _conditional_fit(bases, responses, atoms, active)
        except ValueError:
            return None

    def objective(parameters):
        result = evaluate(parameters)
        if result is None:
            return float("inf")
        _, rss = result
        return float(sum(response.size*np.log(value/response.size)
                         for response, value in zip(responses, rss)))

    global_result = differential_evolution(
        objective, list(bounds), seed=int(seed), polish=False,
        maxiter=int(global_iterations), popsize=10, tol=1e-8, workers=1,
    )
    local_result = minimize(
        objective, global_result.x, method="Nelder-Mead", bounds=list(bounds),
        options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-10},
    )
    parameters = np.asarray(
        local_result.x if local_result.success else global_result.x, dtype=float)
    parameters = np.clip(parameters,
                         [bound[0] for bound in bounds],
                         [bound[1] for bound in bounds])
    coefficients, rss = evaluate(parameters)
    sample_count = sum(response.size for response in responses)
    coefficient_count = sum(base.shape[1]+int(include)
                            for base, include in zip(bases, active))
    parameter_count = coefficient_count+len(bounds)
    bic = objective(parameters)+parameter_count*np.log(sample_count)
    return NetworkProfiledFit(
        parameters=parameters, coefficients=coefficients, equation_rss=rss,
        bic=float(bic), active_equations=active,
        success=bool(global_result.success or local_result.success),
    )


def profiled_multi_mechanism_fit(
    base_matrices: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    atom_builders: Sequence[NetworkAtomBuilder],
    active_equations: Sequence[Sequence[bool]],
    bounds: Sequence[Sequence[tuple[float, float]]],
    *,
    seed: int = 0,
    global_iterations: int = 60,
    initial_parameters: Sequence[float] | None = None,
    use_global_search: bool = True,
    local_iterations: int = 3000,
) -> MultiMechanismProfiledFit:
    """Jointly profile several shared mechanisms and their fixed topologies.

    Each mechanism owns one nonlinear parameter vector shared across equations,
    while every admitted destination edge receives an independent outer
    coefficient.  Candidate-family and topology search remains outside this
    numerical primitive so model-selection evidence is explicit and auditable.
    """
    if not atom_builders:
        raise ValueError("at least one mechanism is required")
    if not (len(atom_builders) == len(active_equations) == len(bounds)):
        raise ValueError("builders, topologies, and bounds must have equal length")
    bases = tuple(np.asarray(matrix, dtype=float) for matrix in base_matrices)
    responses = tuple(np.asarray(target, dtype=float) for target in targets)
    if not bases or len(bases) != len(responses):
        raise ValueError("base matrices and targets must have equal nonzero length")
    for base, response in zip(bases, responses):
        if (base.ndim != 2 or response.ndim != 1
                or base.shape[0] != response.size
                or not np.all(np.isfinite(base))
                or not np.all(np.isfinite(response))):
            raise ValueError("each finite base matrix must match its target")

    topologies = tuple(tuple(bool(value) for value in topology)
                       for topology in active_equations)
    parameter_bounds = tuple(tuple(tuple(value) for value in local_bounds)
                             for local_bounds in bounds)
    for topology, local_bounds in zip(topologies, parameter_bounds):
        if len(topology) != len(responses) or not any(topology):
            raise ValueError("each mechanism needs a nonempty equation topology")
        if not local_bounds or any(low >= high for low, high in local_bounds):
            raise ValueError("each mechanism needs valid nonlinear bounds")

    sizes = [len(local_bounds) for local_bounds in parameter_bounds]
    offsets = np.cumsum([0, *sizes])
    flat_bounds = [item for local_bounds in parameter_bounds for item in local_bounds]
    rss_floors = tuple(
        np.finfo(float).eps * max(float(response @ response), 1.0)
        for response in responses
    )
    if not use_global_search and initial_parameters is None:
        raise ValueError("local-only multi-mechanism fitting requires initial parameters")
    if initial_parameters is not None:
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.shape != (len(flat_bounds),) or not np.all(np.isfinite(initial)):
            raise ValueError("initial parameters must be one finite flat vector")
        initial = np.clip(
            initial,
            [bound[0] for bound in flat_bounds],
            [bound[1] for bound in flat_bounds],
        )
    else:
        initial = None

    def split(parameters):
        vector = np.asarray(parameters, dtype=float)
        return tuple(vector[offsets[index]:offsets[index + 1]]
                     for index in range(len(sizes)))

    def evaluate(parameters):
        local_parameters = split(parameters)
        mechanism_atoms = []
        for builder, values in zip(atom_builders, local_parameters):
            atoms = tuple(np.asarray(atom, dtype=float) for atom in builder(values))
            if (len(atoms) != len(responses)
                    or any(atom.shape != response.shape or not np.all(np.isfinite(atom))
                           for atom, response in zip(atoms, responses))):
                return None
            mechanism_atoms.append(atoms)
        coefficients, rss = [], []
        for equation, (base, response) in enumerate(zip(bases, responses)):
            columns = [base]
            for atoms, topology in zip(mechanism_atoms, topologies):
                if topology[equation]:
                    columns.append(atoms[equation][:, None])
            matrix = np.column_stack(columns)
            estimate = np.linalg.lstsq(matrix, response, rcond=None)[0]
            residual = response - matrix @ estimate
            coefficients.append(estimate)
            rss.append(max(float(residual @ residual), np.finfo(float).tiny))
        return tuple(coefficients), np.asarray(rss)

    def objective(parameters):
        result = evaluate(parameters)
        if result is None:
            return float("inf")
        _, rss = result
        return float(sum(
            response.size * np.log(max(value, floor) / response.size)
            for response, value, floor in zip(responses, rss, rss_floors)
        ))

    global_result = None
    if use_global_search:
        global_result = differential_evolution(
            objective, flat_bounds, seed=int(seed), polish=False,
            maxiter=int(global_iterations), popsize=10, tol=1e-8, workers=1,
        )
        if initial is None or objective(global_result.x) < objective(initial):
            initial = np.asarray(global_result.x, dtype=float)
    local_result = minimize(
        objective, initial, method="Nelder-Mead", bounds=flat_bounds,
        options={"maxiter": int(local_iterations), "xatol": 1e-9, "fatol": 1e-10},
    )
    local_values = np.asarray(local_result.x, dtype=float)
    flat_parameters = np.asarray(
        local_values
        if np.all(np.isfinite(local_values))
        and objective(local_values) <= objective(initial)
        else initial,
        dtype=float,
    )
    flat_parameters = np.clip(
        flat_parameters,
        [bound[0] for bound in flat_bounds],
        [bound[1] for bound in flat_bounds],
    )
    coefficients, rss = evaluate(flat_parameters)
    sample_count = sum(response.size for response in responses)
    coefficient_count = sum(
        base.shape[1] + sum(topology[equation] for topology in topologies)
        for equation, base in enumerate(bases)
    )
    parameter_count = coefficient_count + len(flat_bounds)
    bic = objective(flat_parameters) + parameter_count * np.log(sample_count)
    return MultiMechanismProfiledFit(
        parameters=split(flat_parameters),
        coefficients=coefficients,
        equation_rss=rss,
        bic=float(bic),
        active_equations=topologies,
        success=bool(
            local_result.success
            or (global_result is not None and global_result.success)
        ),
    )


def _canonical_parameter_groups(groups, equation_count):
    canonical = []
    for mechanism_groups in groups:
        if len(mechanism_groups) != equation_count:
            raise ValueError("each parameter grouping must match the equation count")
        relabel, next_label, row = {}, 0, []
        for label in mechanism_groups:
            if label is None:
                row.append(None)
                continue
            if not isinstance(label, (int, np.integer)) or int(label) < 0:
                raise ValueError("active parameter-group labels must be nonnegative integers")
            label = int(label)
            if label not in relabel:
                relabel[label] = next_label
                next_label += 1
            row.append(relabel[label])
        if next_label == 0:
            raise ValueError("each mechanism must enter at least one equation")
        canonical.append(tuple(row))
    return tuple(canonical)


def profiled_partitioned_mechanism_fit(
    base_matrices: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    atom_builders: Sequence[NetworkAtomBuilder],
    parameter_groups: Sequence[Sequence[int | None]],
    bounds: Sequence[Sequence[tuple[float, float]]],
    *,
    seed: int = 0,
    global_iterations: int = 60,
    initial_parameters: Sequence[float] | None = None,
    use_global_search: bool = True,
    local_iterations: int = 3000,
    scale_local_parameters: bool = False,
    local_method: str = "Nelder-Mead",
) -> PartitionedMechanismProfiledFit:
    """Profile mechanisms under a candidate equation-sharing partition.

    A singleton destination has no cross-equation sharing.  Repeated group
    labels impose sharing only on the named destination equations, while
    distinct labels fit independent parameter vectors even when the functional
    family is the same.  Every active destination retains its own outer
    coefficient.  BIC counts one inner vector per sharing group, preventing a
    more flexible independent model from winning without enough residual gain.
    """
    if not atom_builders:
        raise ValueError("at least one mechanism is required")
    if not (len(atom_builders) == len(parameter_groups) == len(bounds)):
        raise ValueError("builders, parameter groups, and bounds must have equal length")
    bases = tuple(np.asarray(matrix, dtype=float) for matrix in base_matrices)
    responses = tuple(np.asarray(target, dtype=float) for target in targets)
    if not bases or len(bases) != len(responses):
        raise ValueError("base matrices and targets must have equal nonzero length")
    for base, response in zip(bases, responses):
        if (base.ndim != 2 or response.ndim != 1
                or base.shape[0] != response.size
                or not np.all(np.isfinite(base))
                or not np.all(np.isfinite(response))):
            raise ValueError("each finite base matrix must match its target")

    groups = _canonical_parameter_groups(parameter_groups, len(responses))
    # Exact synthetic controls can drive RSS into round-off.  Below this
    # response-scaled machine-precision floor, log(RSS) differences have no
    # physical meaning and otherwise cause very long optimizer tails.
    rss_floors = tuple(
        np.finfo(float).eps * max(float(response @ response), 1.0)
        for response in responses
    )
    parameter_bounds = tuple(tuple(tuple(value) for value in local_bounds)
                             for local_bounds in bounds)
    for local_bounds in parameter_bounds:
        if not local_bounds or any(low >= high for low, high in local_bounds):
            raise ValueError("each mechanism needs valid nonlinear bounds")

    group_counts = tuple(max(label for label in row if label is not None) + 1
                         for row in groups)
    blocks = []
    flat_bounds = []
    offset = 0
    for mechanism, (group_count, local_bounds) in enumerate(
            zip(group_counts, parameter_bounds)):
        mechanism_blocks = []
        for group in range(group_count):
            start = offset
            flat_bounds.extend(local_bounds)
            offset += len(local_bounds)
            mechanism_blocks.append((start, offset))
        blocks.append(tuple(mechanism_blocks))
    blocks = tuple(blocks)

    if not use_global_search and initial_parameters is None:
        raise ValueError("local-only partition fitting requires initial parameters")
    if initial_parameters is not None:
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.shape != (len(flat_bounds),) or not np.all(np.isfinite(initial)):
            raise ValueError("initial parameters must be one finite flat vector")
        initial = np.clip(
            initial,
            [bound[0] for bound in flat_bounds],
            [bound[1] for bound in flat_bounds],
        )
    else:
        initial = None

    def split(parameters):
        vector = np.asarray(parameters, dtype=float)
        return tuple(tuple(vector[start:stop] for start, stop in mechanism_blocks)
                     for mechanism_blocks in blocks)

    def evaluate(parameters):
        local_parameters = split(parameters)
        mechanism_atoms = []
        for builder, group_parameters in zip(atom_builders, local_parameters):
            atoms_by_group = []
            for values in group_parameters:
                atoms = tuple(np.asarray(atom, dtype=float) for atom in builder(values))
                if (len(atoms) != len(responses)
                        or any(atom.shape != response.shape
                               or not np.all(np.isfinite(atom))
                               for atom, response in zip(atoms, responses))):
                    return None
                atoms_by_group.append(atoms)
            mechanism_atoms.append(tuple(atoms_by_group))

        coefficients, rss = [], []
        for equation, (base, response) in enumerate(zip(bases, responses)):
            columns = [base]
            for atoms_by_group, mechanism_groups in zip(mechanism_atoms, groups):
                group = mechanism_groups[equation]
                if group is not None:
                    columns.append(atoms_by_group[group][equation][:, None])
            matrix = np.column_stack(columns)
            estimate = np.linalg.lstsq(matrix, response, rcond=None)[0]
            residual = response - matrix @ estimate
            coefficients.append(estimate)
            rss.append(max(float(residual @ residual), np.finfo(float).tiny))
        return tuple(coefficients), np.asarray(rss)

    def objective(parameters):
        result = evaluate(parameters)
        if result is None:
            return float("inf")
        _, rss = result
        return float(sum(
            response.size * np.log(max(value, floor) / response.size)
            for response, value, floor in zip(responses, rss, rss_floors)
        ))

    global_result = None
    if use_global_search:
        global_result = differential_evolution(
            objective, flat_bounds, seed=int(seed), polish=False,
            maxiter=int(global_iterations), popsize=10, tol=1e-8, workers=1,
        )
        if initial is None or objective(global_result.x) < objective(initial):
            initial = np.asarray(global_result.x, dtype=float)
    lower = np.asarray([bound[0] for bound in flat_bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in flat_bounds], dtype=float)
    if scale_local_parameters:
        local_start = (np.asarray(initial, dtype=float) - lower) / (upper - lower)
        local_bounds = [(0.0, 1.0)] * len(flat_bounds)

        def local_objective(values):
            return objective(lower + np.asarray(values, dtype=float) * (upper - lower))
    else:
        local_start = initial
        local_bounds = flat_bounds
        local_objective = objective
    if local_method not in {"Nelder-Mead", "L-BFGS-B", "Powell"}:
        raise ValueError("unsupported local optimization method")
    local_options = (
        {"maxiter": int(local_iterations), "xatol": 1e-9, "fatol": 1e-10}
        if local_method == "Nelder-Mead"
        else {"maxiter": int(local_iterations), "ftol": 1e-12, "gtol": 1e-9}
        if local_method == "L-BFGS-B"
        else {"maxiter": int(local_iterations), "xtol": 1e-9, "ftol": 1e-12}
    )
    local_result = minimize(
        local_objective, local_start, method=local_method, bounds=local_bounds,
        options=local_options,
    )
    local_values = np.asarray(local_result.x, dtype=float)
    if scale_local_parameters:
        local_values = lower + local_values * (upper - lower)
    flat_parameters = np.asarray(
        local_values
        if np.all(np.isfinite(local_values))
        and objective(local_values) <= objective(initial)
        else initial,
        dtype=float,
    )
    flat_parameters = np.clip(
        flat_parameters,
        [bound[0] for bound in flat_bounds],
        [bound[1] for bound in flat_bounds],
    )
    coefficients, rss = evaluate(flat_parameters)
    sample_count = sum(response.size for response in responses)
    coefficient_count = sum(
        base.shape[1] + sum(row[equation] is not None for row in groups)
        for equation, base in enumerate(bases)
    )
    inner_parameter_count = sum(
        group_count * len(local_bounds)
        for group_count, local_bounds in zip(group_counts, parameter_bounds)
    )
    bic = (objective(flat_parameters)
           + (coefficient_count + inner_parameter_count) * np.log(sample_count))
    return PartitionedMechanismProfiledFit(
        parameters=split(flat_parameters), coefficients=coefficients,
        equation_rss=rss, bic=float(bic), parameter_groups=groups,
        success=bool(
            local_result.success
            or (global_result is not None and global_result.success)
        ),
    )
