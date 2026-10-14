"""Observation-only cached-forward adapter for the unified decision prototype.

No truth trajectory or hidden state is accepted by this API. Covariance and
decision limits must come from an external, declared calibration protocol.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from .unified import WeakProposal, reconcile_identification


@dataclass(frozen=True)
class ObservationPacket:
    values: np.ndarray
    covariance: np.ndarray
    labels: tuple[str, ...]
    units: tuple[str, ...]
    operator_id: str
    condition_id: str
    geometry_convention_id: str | None = None

    def __post_init__(self):
        values = np.array(self.values, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        labels, units = tuple(self.labels), tuple(self.units)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError("observations must be a nonempty finite vector")
        if covariance.shape != (values.size, values.size) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be finite and observation-aligned")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive definite") from exc
        if len(labels) != values.size or len(set(labels)) != len(labels):
            raise ValueError("observation labels must be aligned and unique")
        if len(units) != values.size or not all(isinstance(s, str) and s for s in (*labels, *units)):
            raise ValueError("labels and explicit units must be nonempty strings")
        if not self.operator_id or not self.condition_id:
            raise ValueError("observation operator and condition identifiers are required")
        if self.geometry_convention_id is not None and (not isinstance(self.geometry_convention_id,str) or not self.geometry_convention_id):
            raise ValueError("geometry convention must be a nonempty identifier")
        values.setflags(write=False); covariance.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "units", units)

    @property
    def observation_id(self):
        metadata = json.dumps(dict(labels=self.labels, units=self.units,
                                   operator=self.operator_id, condition=self.condition_id),
                              ensure_ascii=False, sort_keys=True).encode("utf-8")
        # Preserve legacy packet hashes when no convention was declared.
        if self.geometry_convention_id is not None:
            metadata += json.dumps(dict(geometry_convention=self.geometry_convention_id),sort_keys=True).encode()
        digest = hashlib.sha256(metadata)
        digest.update(self.values.astype("<f8").tobytes())
        digest.update(self.covariance.astype("<f8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class ForwardTemplate:
    candidate_id: str
    nuisance_id: str
    prediction: np.ndarray
    labels: tuple[str, ...]
    units: tuple[str, ...]
    operator_id: str
    condition_id: str
    geometry_convention_id: str | None = None
    geometry_realization_id: str | None = None

    def __post_init__(self):
        prediction = np.array(self.prediction, dtype=float, copy=True)
        if prediction.ndim != 1 or not prediction.size or not np.all(np.isfinite(prediction)):
            raise ValueError("template prediction must be a finite vector")
        if not self.candidate_id or not self.nuisance_id:
            raise ValueError("candidate and nuisance identifiers are required")
        if (self.geometry_convention_id is None)!=(self.geometry_realization_id is None):
            raise ValueError("template geometry convention and realization must be declared together")
        if self.geometry_convention_id is not None and not all(isinstance(v,str) and v for v in
            (self.geometry_convention_id,self.geometry_realization_id)):
            raise ValueError("nonempty geometry identifiers required")
        prediction.setflags(write=False)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "units", tuple(self.units))


def profile_cached_templates(packet: ObservationPacket, candidate_ids: Sequence[str],
                             templates: Sequence[ForwardTemplate]):
    """Minimum mean whitened squared residual over discrete nuisance templates.

This implements a discrete template profile, not R169 continuous geometry or
material interpolation. Its thresholds cannot be borrowed across protocols.
"""
    ids = tuple(candidate_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("candidate universe must be nonempty and unique")
    scores, selected = {}, {}
    factor = np.linalg.cholesky(packet.covariance)
    seen = set()
    for template in templates:
        if template.geometry_convention_id != packet.geometry_convention_id:
            raise ValueError("template and observations use incompatible geometry conventions")
        if template.candidate_id not in ids:
            raise ValueError("template has an undeclared candidate")
        token = (template.candidate_id, template.nuisance_id)
        if token in seen:
            raise ValueError("duplicate candidate nuisance template")
        seen.add(token)
        if (template.labels != packet.labels or template.units != packet.units
                or template.operator_id != packet.operator_id
                or template.condition_id != packet.condition_id
                or template.prediction.shape != packet.values.shape):
            raise ValueError("template and observations have incompatible physical channels")
        residual = np.linalg.solve(factor, packet.values - template.prediction)
        score = float(np.mean(residual ** 2))
        if template.candidate_id not in scores or score < scores[template.candidate_id]:
            scores[template.candidate_id] = score
            selected[template.candidate_id] = template.nuisance_id
    return scores, selected


def identify_observation_packet(
    packet: ObservationPacket, *, candidate_ids: Sequence[str],
    templates: Sequence[ForwardTemplate], gap_threshold: float,
    calibration_id: str, adequacy_upper_bound: float | None,
    weak_adapter: Callable[[ObservationPacket], WeakProposal] | None = None,
):
    """Run a same-packet optional proposal and full-universe forward evaluation.

Weak adapter is not supplied by default: existing full-field routines require
additional spatially distributed observations and a physical library mapping.
"""
    proposal = None if weak_adapter is None else weak_adapter(packet)
    if proposal is not None and not isinstance(proposal, WeakProposal):
        raise TypeError("weak adapter must return an explicit WeakProposal")
    geometry_profile_complete = None
    if proposal is not None:
        if proposal.geometry_convention_id != packet.geometry_convention_id:
            raise ValueError("weak proposal and packet use incompatible geometry conventions")
        if packet.geometry_convention_id is not None:
            declared = proposal.geometry_realization_ids
            if any(not isinstance(v,str) or not v for v in declared) or len(set(declared))!=len(declared):
                raise ValueError("weak geometry realizations must be nonempty and unique")
            forward_geometries={v.geometry_realization_id for v in templates}
            if set(declared)-forward_geometries:
                raise ValueError("weak proposal uses geometry outside forward template profile")
            geometry_profile_complete=bool(forward_geometries) and forward_geometries.issubset(declared)
            if proposal.complete and not geometry_profile_complete:
                proposal=replace(proposal,complete=False)
    scores, selected = profile_cached_templates(packet, candidate_ids, templates)
    result = reconcile_identification(
        observation_id=packet.observation_id, candidate_ids=candidate_ids,
        forward_scores=scores, gap_threshold=gap_threshold,
        calibration_id=calibration_id, adequacy_upper_bound=adequacy_upper_bound,
        proposal=proposal,
        geometry_convention_id=packet.geometry_convention_id,
    )
    result.update(selected_nuisance_by_candidate=selected,
                  selected_geometry_by_candidate={c:next(v.geometry_realization_id for v in templates
                      if v.candidate_id==c and v.nuisance_id==n) for c,n in selected.items()},
                  geometry_declaration_is_not_physical_validation=True,
                  geometry_alignment=("declared_convention_matched" if packet.geometry_convention_id is not None
                                      else "undeclared_legacy_not_verified"),
                  weak_geometry_profile_complete=geometry_profile_complete,
                  weak_proposal_effectively_complete=None if proposal is None else proposal.complete,
                  weak_stage_executed=weak_adapter is not None,
                  profile_kind="discrete_cached_template",
                  score_kind="mean_whitened_squared_residual")
    return result
