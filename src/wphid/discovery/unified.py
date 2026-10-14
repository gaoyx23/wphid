"""Conservative decision bridge, not a validated unified physical estimator.

Adapters must consume the same observation packet. This module deliberately
does not read truth trajectories, select calibration thresholds, or reconstruct
unobserved fields. Full-library forward scoring protects against proposal loss.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WeakProposal:
    observation_id: str
    candidate_ids: tuple[str, ...]
    complete: bool
    provenance: str
    geometry_convention_id: str | None = None
    geometry_realization_ids: tuple[str, ...] = ()


def reconcile_identification(
    *, observation_id: str, candidate_ids: Sequence[str],
    forward_scores: Mapping[str, float], gap_threshold: float,
    calibration_id: str, adequacy_upper_bound: float | None,
    proposal: WeakProposal | None = None,
    parameter_estimates: Mapping[str, Mapping] | None = None,
    parameter_identifiable: bool = False,
    geometry_convention_id: str | None = None,
) -> dict:
    """Combine same-packet evidence without silently intersecting two sets.

An independently calibrated forward set is never narrowed by a proposal.
Conflicting complete proposals expand the set and withhold parameters; this
conservative policy needs fresh scientific calibration before publication.
Missing/nonfinite scores or absent adequacy calibration withhold interpretation.
"""
    ids = tuple(candidate_ids)
    if not observation_id or not calibration_id:
        raise ValueError("observation and calibration identifiers are required")
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(c, str) or not c for c in ids):
        raise ValueError("candidate identifiers must be nonempty and unique")
    if not math.isfinite(gap_threshold) or gap_threshold < 0:
        raise ValueError("gap threshold must be finite and nonnegative")
    if adequacy_upper_bound is not None and (
        not math.isfinite(adequacy_upper_bound) or adequacy_upper_bound < 0
    ):
        raise ValueError("adequacy upper bound must be finite and nonnegative")
    if set(forward_scores) - set(ids):
        raise ValueError("scores contain undeclared candidates")
    if proposal is not None:
        if proposal.observation_id != observation_id:
            raise ValueError("proposal and forward evidence use different observations")
        if proposal.geometry_convention_id != geometry_convention_id:
            raise ValueError("proposal and forward geometry conventions differ or are undeclared")
        if not proposal.provenance:
            raise ValueError("proposal provenance is required")
        if len(set(proposal.candidate_ids)) != len(proposal.candidate_ids):
            raise ValueError("proposal candidates must be unique")
        if set(proposal.candidate_ids) - set(ids):
            raise ValueError("proposal requires an explicit library expansion")
    scores = {c: float(v) for c, v in forward_scores.items()}
    complete = set(scores) == set(ids) and all(math.isfinite(v) and v >= 0 for v in scores.values())
    result = {
        "schema_version": "unified-decision-v0",
        "observation_id": observation_id,
        "geometry_convention_id": geometry_convention_id,
        "calibration_id": calibration_id,
        "candidate_universe": list(ids),
        "forward_scores": scores,
        "forward_complete": complete,
        "gap_threshold": gap_threshold,
        "zero_gap_hard_ranking": gap_threshold == 0,
        "adequacy_upper_bound": adequacy_upper_bound,
        "proposal_provenance": None if proposal is None else proposal.provenance,
        "parameter_output": None,
        "parameter_identifiable": bool(parameter_identifiable),
        "scientific_confirmation": False,
    }
    if not complete:
        return dict(result, status="incomplete_forward", candidate_set=list(ids), forward_set=None)
    best = min(scores.values())
    forward = tuple(c for c in ids if scores[c] - best <= gap_threshold)
    result.update(best_score=best, forward_set=list(forward))
    if adequacy_upper_bound is None:
        return dict(result, status="adequacy_uncalibrated", candidate_set=list(ids))
    if best > adequacy_upper_bound:
        return dict(result, status="model_inadequate", candidate_set=[], library_review_required=True)
    conflict = proposal is not None and proposal.complete and (
        not proposal.candidate_ids or not set(forward).issubset(proposal.candidate_ids)
    )
    if conflict:
        combined = set(forward) | set(proposal.candidate_ids)
        return dict(result, status="stage_conflict", candidate_set=[c for c in ids if c in combined])
    result.update(status="unique" if len(forward) == 1 else "candidate_set", candidate_set=list(forward))
    if len(forward) == 1 and parameter_identifiable and parameter_estimates is not None:
        result["parameter_output"] = parameter_estimates.get(forward[0])
    return result
