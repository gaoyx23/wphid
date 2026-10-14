"""Independent component solver with one program-wide resource envelope.

One-component and null candidates delegate to immutable V1. Disconnected
candidates share the same declared call and CPU caps, apportioned by dimension
after one feasible baseline. Quotas are not recycled or multiplied by blocks.
Saved coefficients are assembled without a final coefficient refit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import process_time, perf_counter

import numpy as np
from scipy.optimize import differential_evolution, minimize

from wphid.final_candidate_v1.solver import (
    BudgetExceeded, FitBudget, FittedCandidate, conditional_profile, fit_candidate)
from .factorization import factorize, prepare_profile, component_profile


@dataclass(frozen=True)
class FactoredFit:
    fit: FittedCandidate
    factorization: object
    audit: dict


def allocate_calls(total: int, dimensions: tuple[int, ...]) -> tuple[int, ...]:
    """Largest-remainder integer apportionment; ties follow canonical order."""
    if total < 0 or not dimensions or any(d < 1 for d in dimensions):
        raise ValueError("Nonnegative calls and positive component dimensions required")
    width = sum(dimensions)
    allocation = [total*d//width for d in dimensions]
    order = sorted(range(len(dimensions)), key=lambda i: (-(total*dimensions[i] % width), i))
    for i in order[:total-sum(allocation)]:
        allocation[i] += 1
    return tuple(allocation)


def fit_factored_candidate(view, candidate, budget=FitBudget()) -> FactoredFit:
    started_cpu, started_wall = process_time(), perf_counter()
    if tuple(c.name for c in view.conditions) != view.scale.condition_names:
        raise ValueError("Estimator view must use the frozen training conditions")
    layout = factorize(candidate, len(view.scale.base_norms))
    if len(layout.components) <= 1:
        remaining = budget.maximum_cpu_seconds-(process_time()-started_cpu)
        if remaining <= 0:
            raise BudgetExceeded("Program CPU cap exhausted before delegated fit")
        fit = fit_candidate(view, candidate, replace(budget, maximum_cpu_seconds=remaining))
        fit = replace(fit, cpu_seconds=process_time()-started_cpu, wall_seconds=perf_counter()-started_wall)
        return FactoredFit(fit, layout, {
            "mode": "immutable_V1_delegate", "component_count": len(layout.components),
            "maximum_program_calls": budget.maximum_objective_calls,
            "maximum_program_CPU_seconds": budget.maximum_cpu_seconds,
            "program_objective_calls": fit.objective_calls,
            "component_records": [], "final_coefficient_refits": 0,
            "extra_baseline_calls": 0, "quota_redistribution": False})

    problem = prepare_profile(view, candidate)
    bounds = np.asarray(candidate.flat_bounds, dtype=np.float64)
    lower, span = bounds[:, 0], bounds[:, 1]-bounds[:, 0]
    parameters = lower+budget.initial_fractions[0]*span
    if process_time()-started_cpu >= budget.maximum_cpu_seconds:
        raise BudgetExceeded("Program CPU cap exhausted before first feasible point")
    coefficients, equation_rss = conditional_profile(view, candidate, parameters)
    coefficients, equation_rss = list(coefficients), list(equation_rss)
    calls = 1
    dimensions = tuple(len(c.parameter_indices) for c in layout.components)
    call_quotas = allocate_calls(budget.maximum_objective_calls-1, dimensions)
    remaining_cpu = max(0., budget.maximum_cpu_seconds-(process_time()-started_cpu))
    cpu_quotas = tuple(remaining_cpu*d/sum(dimensions) for d in dimensions)
    records, restarts = [], []

    for index, component in enumerate(layout.components):
        indices = list(component.parameter_indices)
        part_lower, part_span = lower[indices], span[indices]
        component_started = process_time()
        component_calls = 0
        baseline_rss = tuple(equation_rss[e] for e in component.equations)
        baseline_value = float(sum(problem.convention.row_counts[e]*np.log(
            max(r, problem.convention.floors[e])/problem.convention.row_counts[e])
            for e, r in zip(component.equations, baseline_rss)))
        best = (baseline_value, parameters[indices].copy(),
                tuple(coefficients[e].copy() for e in component.equations), baseline_rss)

        def check_budget():
            if (calls >= budget.maximum_objective_calls
                    or process_time()-started_cpu >= budget.maximum_cpu_seconds
                    or component_calls >= call_quotas[index]
                    or process_time()-component_started >= cpu_quotas[index]):
                raise BudgetExceeded("Program or predeclared component quota exhausted")

        def objective(values):
            nonlocal calls, component_calls, best
            check_budget()
            calls += 1
            component_calls += 1
            coefs, errors, value = component_profile(problem, component, values)
            if value < best[0]:
                best = (value, np.asarray(values).copy(), coefs, errors)
            return value

        quota_exhausted, component_converged = False, False
        for start_index, (seed, fraction) in enumerate(zip(budget.seeds, budget.initial_fractions)):
            restart = {"component": index, "seed": seed, "initial_fraction": fraction,
                       "global_success": None, "local_success": None,
                       "budget_exhausted": False, "baseline_start_reused": start_index == 0}
            before = component_calls
            try:
                check_budget()
                start = part_lower+fraction*part_span
                initial_value = baseline_value if start_index == 0 else objective(start)
                if budget.global_iterations:
                    global_fit = differential_evolution(
                        objective, [tuple(b) for b in bounds[indices]], seed=seed, polish=False,
                        maxiter=budget.global_iterations, popsize=budget.population_size,
                        tol=1e-8, workers=1, updating="immediate")
                    restart["global_success"] = bool(global_fit.success)
                    if global_fit.fun < initial_value:
                        start = global_fit.x
                local_fit = minimize(lambda u: objective(part_lower+u*part_span),
                                     (start-part_lower)/part_span, method="Nelder-Mead",
                                     bounds=[(0., 1.)]*len(indices),
                                     options={"maxiter": budget.local_iterations,
                                              "xatol": 1e-9, "fatol": 1e-10})
                restart["local_success"] = bool(local_fit.success)
                restart["terminal_component_objective"] = float(local_fit.fun)
                component_converged |= bool(local_fit.success or restart["global_success"])
            except BudgetExceeded:
                restart["budget_exhausted"] = True
                quota_exhausted = True
            restart["objective_calls"] = component_calls-before
            restarts.append(restart)
            if quota_exhausted:
                break
        parameters[indices] = best[1]
        for e, coef, error in zip(component.equations, best[2], best[3]):
            coefficients[e], equation_rss[e] = coef, error
        records.append({"component": index, "equations": list(component.equations),
                        "parameter_indices": indices, "dimension": dimensions[index],
                        "maximum_objective_calls": call_quotas[index],
                        "maximum_CPU_seconds": cpu_quotas[index],
                        "objective_calls": component_calls,
                        "CPU_seconds": process_time()-component_started,
                        "quota_exhausted": quota_exhausted,
                        "converged_search": component_converged,
                        "baseline_objective": baseline_value,
                        "best_objective": best[0]})

    objective_value = problem.convention.objective(equation_rss)
    score = problem.convention.score(equation_rss, problem.nominal_parameter_count)
    cpu_seconds, wall_seconds = process_time()-started_cpu, perf_counter()-started_wall
    global_cpu_reached = cpu_seconds >= budget.maximum_cpu_seconds
    global_calls_reached = calls >= budget.maximum_objective_calls
    status = ("budget_exhausted" if global_cpu_reached or global_calls_reached
              else "component_quota_exhausted" if any(r["quota_exhausted"] for r in records)
              else "completed_with_converged_components" if all(r["converged_search"] for r in records)
              else "completed_unconverged")
    fit = FittedCandidate(candidate, parameters.copy(), tuple(coefficients), tuple(equation_rss),
                         objective_value, score, problem.nominal_parameter_count,
                         view.scale.fingerprint, status, calls, cpu_seconds, wall_seconds, tuple(restarts))
    return FactoredFit(fit, layout, {
        "mode": "candidate_connectivity_components", "component_count": len(layout.components),
        "maximum_program_calls": budget.maximum_objective_calls,
        "maximum_program_CPU_seconds": budget.maximum_cpu_seconds,
        "program_objective_calls": calls, "baseline_objective_calls": 1,
        "baseline_initial_fraction": budget.initial_fractions[0],
        "call_quota_sum": sum(call_quotas), "CPU_quota_sum": sum(cpu_quotas),
        "quota_rule": "remaining calls/CPU proportional to dimension; integer largest remainder, canonical ties",
        "quota_redistribution": False, "component_records": records,
        "global_CPU_cap_reached": global_cpu_reached, "global_call_cap_reached": global_calls_reached,
        "final_coefficient_refits": 0,
        "CPU_limit_check": "Between objective calls; record actual one-call overshoot and finalization cost",
        "new_observations_or_reweighting": False})
