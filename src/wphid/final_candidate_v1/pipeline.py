"""Integrated truth-free candidate workflow and conditional comparison API.

Reporting thresholds are declared diagnostic candidates, not inherited H3
qualification, calibrated error probabilities, or actual precision guarantees.
"""

from dataclasses import asdict

import numpy as np

from .problem import TrainingScale, validate_split
from .solver import (Candidate, Mechanism, FitBudget, conditional_profile, fit_candidate,
                     physical_coefficients, predict_frozen, search)
from .adam_adapter import AdamBudget, fit_adam


def fit_record(view, fit):
    parameters = fit.candidate.split(fit.parameters)
    return {"hypothesis": [{"mechanism": m.name, "family": m.family.name, "groups": list(m.groups),
                            "internal_parameters_by_group": [p.tolist() for p in block]}
                           for m, block in zip(fit.candidate.mechanisms, parameters)],
            "physical_coefficients": [c.tolist() for c in physical_coefficients(view, fit)],
            "scaled_coefficients": [c.tolist() for c in fit.coefficients],
            "equation_training_rss": list(fit.equation_rss), "training_objective": fit.training_objective,
            "common_final_score": fit.final_score, "nominal_parameter_count": fit.nominal_parameter_count,
            "training_scale_sha256": fit.training_scale_sha256, "optimizer_status": fit.status,
            "objective_calls": fit.objective_calls, "cpu_seconds": fit.cpu_seconds,
            "wall_seconds": fit.wall_seconds, "restarts": list(fit.restarts),
            "actual_parameter_precision": None, "scientific_qualification_established": False}


def prepare_views(conditions, families, train_names, heldout_names):
    conditions = tuple(conditions)
    known = tuple(c.name for c in conditions)
    validate_split(known, train_names, heldout_names)
    by_name = {c.name: c for c in conditions}
    scale = TrainingScale.fit(tuple(by_name[n] for n in train_names), tuple(families))
    return scale.apply(tuple(by_name[n] for n in train_names)), scale.apply(tuple(by_name[n] for n in heldout_names))


def frozen_residual_report(view, fit):
    predictions = predict_frozen(view, fit)
    targets = tuple(np.concatenate([c.targets[e] for c in view.conditions]) for e in range(len(predictions)))
    return {"conditions": [c.name for c in view.conditions], "refitted": False,
            "prediction_kind": "weak_target_given_observed_input_states_not_free_trajectory",
            "equation_weak_target_relative_residual": [float(np.linalg.norm(y-p)/max(np.linalg.norm(y), 1e-12))
                                                       for y, p in zip(targets, predictions)],
            "scaling_source": list(view.scale.condition_names), "scale_sha256": view.scale.fingerprint}


def _structure_key(candidate):
    return tuple((m.name, m.family.name, tuple(g is not None for g in m.groups)) for m in candidate.mechanisms)


def training_diagnostics(training, ranked, *, minimum_margin=2., maximum_amplitude_dispersion=.4,
                         maximum_median_disagreement=.2, minimum_sign_consistency=1.):
    selected = ranked[0]
    alternatives = [f.final_score-selected.final_score for f in ranked[1:]
                    if _structure_key(f.candidate) != _structure_key(selected.candidate)]
    partition_alternatives = [f.final_score-selected.final_score for f in ranked[1:]
                              if _structure_key(f.candidate) == _structure_key(selected.candidate)]
    structure_margin = min(alternatives) if alternatives else None
    partition_margin = min(partition_alternatives) if partition_alternatives else None
    margins_supported = (structure_margin is not None and structure_margin >= minimum_margin
                         and (not selected.candidate.mechanisms or
                              (partition_margin is not None and partition_margin >= minimum_margin)))
    # Condition-wise linear amplitude diagnostics use only training observations,
    # fixing nonlinear parameters. They are refit diagnostics, not prediction.
    per_condition = []
    for condition in training.conditions:
        view = training.scale.apply((condition,))
        coef, _ = conditional_profile(view, selected.candidate, selected.parameters)
        per_condition.append(coef)
    rows, passed = [], True
    pooled = selected.coefficients
    for e, base in enumerate(training.bases):
        offset = base.shape[1]
        for m in selected.candidate.mechanisms:
            if m.groups[e] is None:
                continue
            values = np.asarray([coefs[e][offset] for coefs in per_condition])
            median = float(np.median(values))
            denominator = max(abs(median), 1e-12)
            dispersion = float(np.median(np.abs(values-median))/denominator)
            disagreement = float(abs(pooled[e][offset]-median)/denominator)
            sign_consistency = float(np.mean(np.sign(values) == np.sign(pooled[e][offset])))
            stable = (dispersion <= maximum_amplitude_dispersion
                      and disagreement <= maximum_median_disagreement
                      and sign_consistency >= minimum_sign_consistency)
            rows.append({"equation": e, "mechanism": m.name, "scaled_amplitude_mad_relative_to_median": dispersion,
                         "scaled_amplitude_pooled_median_disagreement": disagreement,
                         "sign_consistency": sign_consistency, "stable": stable})
            passed &= stable
            offset += 1
    tier = "competing_hypotheses"
    if margins_supported:
        tier = "null_candidate" if not selected.candidate.mechanisms else "structure_candidate"
        if selected.candidate.mechanisms and passed and selected.status == "completed_with_converged_search":
            tier = "stable_decomposition_candidate"
    return {"candidate_report_tier": tier, "structure_margin": structure_margin,
            "partition_margin": partition_margin, "training_amplitude_diagnostics": rows,
            "condition_amplitudes_refitted_for_training_diagnostic_only": True,
            "point_estimates_retained_even_if_reporting_checks_fail": True,
            "thresholds": {"minimum_margin": minimum_margin,
                           "maximum_amplitude_dispersion": maximum_amplitude_dispersion,
                           "maximum_median_disagreement": maximum_median_disagreement,
                           "minimum_sign_consistency": minimum_sign_consistency},
            "parameter_precision_proven_by_this_tier": False, "H3_V6_gates_inherited": False,
            "calibrated_confidence": False}


def identify(conditions, candidates, train_names, heldout_names, *, maximum_finalists=12, budget=FitBudget()):
    candidates = tuple(candidates)
    families = {}
    for candidate in candidates:
        for mechanism in candidate.mechanisms:
            family = mechanism.family
            if family.name in families and families[family.name] != family:
                raise ValueError("Same family name has different bounds/reference conventions")
            families[family.name] = family
    if not families:
        raise ValueError("Identification requires a declared family library alongside null")
    training, heldout = prepare_views(conditions, tuple(families.values()), train_names, heldout_names)
    result = search(training, candidates, maximum_finalists=maximum_finalists, budget=budget)
    ranked = result.pop("ranked")
    diagnostics = training_diagnostics(training, ranked)
    return {"workflow": "WPHID_FINAL_CANDIDATE_IMPLEMENTATION_V1", "search": result,
            "training_conditions": list(train_names), "heldout_conditions": list(heldout_names),
            "training_scale_sha256": training.scale.fingerprint, "budget": asdict(budget),
            "ranked_candidates": [fit_record(training, fit) for fit in ranked],
            "reporting": diagnostics, "frozen_heldout": frozen_residual_report(heldout, ranked[0]),
            "legacy_default_changed": False, "scientific_qualification_established": False}


def conditional_pair(conditions, candidate, train_names, heldout_names, *,
                     wphid_budget=FitBudget(), adam_budget=AdamBudget()):
    families = tuple({m.family.name: m.family for m in candidate.mechanisms}.values())
    training, heldout = prepare_views(conditions, families, train_names, heldout_names)
    wphid_fit = fit_candidate(training, candidate, wphid_budget)
    adam_result = fit_adam(training, candidate, adam_budget)
    adam_fit = adam_result.pop("fit")
    return {"scope": "known_family_topology_partition_common_weak_input_conditional_comparison",
            "training_scale_sha256": training.scale.fingerprint,
            "wphid": fit_record(training, wphid_fit), "adam_adapter": fit_record(training, adam_fit),
            "adam_adaptation": adam_result,
            "wphid_frozen_heldout": frozen_residual_report(heldout, wphid_fit),
            "adam_frozen_heldout": frozen_residual_report(heldout, adam_fit),
            "truth_used_in_estimators": False, "parameter_accuracy_compared": False,
            "scientific_qualification_established": False}


def change_one_partition(candidate, mechanism_name, groups):
    """Build the Q2 control without changing family, topology or amplitudes."""
    if not any(m.name == mechanism_name for m in candidate.mechanisms):
        raise ValueError("Unknown mechanism for sharing control")
    result = []
    for m in candidate.mechanisms:
        if m.name != mechanism_name:
            result.append(m)
            continue
        if tuple(g is not None for g in groups) != tuple(g is not None for g in m.groups):
            raise ValueError("Sharing-only control must preserve destination topology")
        result.append(Mechanism(m.name, m.family, tuple(groups)))
    return Candidate(tuple(result))
