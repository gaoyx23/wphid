"""Applicability gate for reference-calibrated after-profile evidence.

Estimating nuisance parameters does not establish calibration uniformly over
their possible true values. Out-of-scope evidence stays advisory, never prunes
the full forward library or authorizes a parameter estimate.
"""
import math


def qualify_profiled_evidence(*, decision, candidate_universe, calibration_id,
        calibrated_prefactor_points, requested_prefactors=None,
        requested_reaction_support="unknown", calibrated_reaction_support="oxygen_and_steam"):
    universe=tuple(candidate_universe)
    if not calibration_id or not universe or len(set(universe))!=len(universe):
        raise ValueError("calibration identity and complete unique universe required")
    conditional=tuple(decision["candidate_set"])
    if not set(conditional)<=set(universe) or set(decision["scores"])!=set(universe):
        raise ValueError("conditional evidence must match complete candidate universe")
    def point(value):
        values=tuple(float(v) for v in value)
        if len(values)!=2 or any(not math.isfinite(v) or v<=0 for v in values):
            raise ValueError("two positive finite reaction prefactors required")
        return values
    points=tuple(point(v) for v in calibrated_prefactor_points)
    if not points:raise ValueError("explicit finite calibration points required")
    requested=None if requested_prefactors is None else point(requested_prefactors)
    reasons=[]
    if requested is None:reasons.append("unknown_prefactors_not_uniformly_calibrated")
    elif requested not in points:reasons.append("requested_prefactors_outside_calibrated_points")
    if requested_reaction_support!=calibrated_reaction_support:
        reasons.append("reaction_support_not_covered_by_conditional_calibration")
    return dict(calibration_id=calibration_id,candidate_universe=list(universe),
        conditional_candidate_set=list(conditional),
        candidate_set=list(universe) if reasons else list(conditional),
        applicability_qualified=not reasons,scope_failures=reasons,
        status="advisory_only_scope_unvalidated" if reasons else "conditional_reference_evidence",
        weak_proposal_complete=not reasons,hard_pruning_authorized=False,
        parameter_output=None,parameter_confidence_intervals=None,
        uniform_unknown_parameter_recovery=False,
        calibrated_prefactor_points=points,requested_prefactors=requested,
        requested_reaction_support=requested_reaction_support)
