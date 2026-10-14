"""Local coupled-forward interpolation on an explicit kinetic triangle.

This is a surrogate, not a new PDE solution. Off-anchor discrepancy must be
validated independently. All candidates see the same measured pressure packet.
"""
import numpy as np


def inverse_noise(values,modes):
    array=np.asarray(values,dtype=float);u=np.asarray(modes,dtype=float)
    return array-u@np.linalg.solve(np.eye(u.shape[1])+u.T@u,u.T@array)


def build_forward_template(*,reference,oxygen_anchor,steam_anchor,scale,modes):
    baseline=np.asarray(reference,dtype=float).reshape(-1)
    sigma=np.asarray(scale,dtype=float).reshape(-1)
    if sigma.shape!=baseline.shape or np.any(sigma<=0) or not np.all(np.isfinite(sigma)):
        raise ValueError("positive common observation scaling required")
    anchors=[np.asarray(v,dtype=float).reshape(-1) for v in (oxygen_anchor,steam_anchor)]
    if any(v.shape!=baseline.shape or not np.all(np.isfinite(v)) for v in (baseline,*anchors)):
        raise ValueError("aligned finite full-field templates required")
    u=np.asarray(modes,dtype=float)
    if u.ndim!=2 or u.shape[0]!=len(baseline) or not np.all(np.isfinite(u)):raise ValueError("aligned noise modes required")
    design=np.column_stack([(v-baseline)/sigma for v in anchors])
    gram=design.T@inverse_noise(design,u)
    if np.linalg.matrix_rank(gram)!=2:raise ValueError("forward parameter basis rank deficient")
    return dict(baseline=baseline,scale=sigma,modes=u,design=design,gram=gram,
        physical_prefactor_anchors=[[1.,1.],[.8,1.],[1.,1.2]],
        model="affine local surrogate of full seven-field trajectories; gas outputs only")


def score_forward(observed,template):
    values=np.asarray(observed,dtype=float).reshape(-1)
    baseline=np.asarray(template["baseline"]);sigma=np.asarray(template["scale"])
    if values.shape!=baseline.shape or not np.all(np.isfinite(values)):raise ValueError("aligned measured pressure packet required")
    y=(values-baseline)/sigma;a=np.asarray(template["design"]);u=np.asarray(template["modes"])
    gram=np.asarray(template["gram"]);rhs=a.T@inverse_noise(y,u)
    trial=[np.array([0.,0.]),np.array([1.,0.]),np.array([0.,1.])]
    unconstrained=np.linalg.solve(gram,rhs)
    if np.all(unconstrained>=0) and unconstrained.sum()<=1:trial.append(unconstrained)
    for i in range(2):
        value=np.zeros(2);value[i]=np.clip(rhs[i]/gram[i,i],0.,1.);trial.append(value)
    start=np.array([0.,1.]);direction=np.array([1.,-1.])
    along=np.clip(direction@(rhs-gram@start)/(direction@gram@direction),0.,1.)
    trial.append(start+along*direction)
    def objective(weights):
        residual=y-a@weights
        return float(residual@inverse_noise(residual,u))
    best=min(trial,key=objective);score=objective(best)
    return dict(weighted_squared_residual=score,triangle_weights=best,
        fitted_prefactor_multipliers=[1-.2*best[0],1+.2*best[1]],
        parameter_output_is_diagnostic_only=True,parameter_confidence_intervals=None)


def reconcile_same_packet(*,universe,forward_set,weak_set,forward_observation_id,
        weak_observation_id,weak_applicability_qualified=False):
    ids=tuple(universe)
    if not ids or len(set(ids))!=len(ids) or not forward_observation_id or forward_observation_id!=weak_observation_id:
        raise ValueError("same packet and unique complete universe required")
    if not set(forward_set)<=set(ids) or not set(weak_set)<=set(ids):raise ValueError("undeclared candidate")
    if not forward_set:
        return dict(status="forward_model_inadequate",candidate_set=[],parameter_output=None,library_review_required=True)
    conflict=weak_applicability_qualified and not set(forward_set)<=set(weak_set)
    output=set(forward_set)|set(weak_set) if conflict else set(forward_set)
    return dict(status="stage_conflict" if conflict else "unique" if len(output)==1 else "candidate_set",
        candidate_set=[n for n in ids if n in output],forward_set=list(forward_set),weak_set=list(weak_set),
        weak_evidence_advisory_only=not weak_applicability_qualified,parameter_output=None,
        proposal_does_not_prune_forward_library=True)
