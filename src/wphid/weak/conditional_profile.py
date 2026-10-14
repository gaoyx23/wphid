"""Parameter-conditioned after-fit residual contrasts, experimental only.

Conditional means use the SAME observation's fitted prefactors. Joint noise is
estimated after that conditioning. Finite anchor calibration is not uniform
validity over a continuous parameter domain or unknown reaction supports.
"""
from math import ceil
import numpy as np
from .empirical_noise import noise_metric,whiten_noise


def _basis(features,center,scale):
    values=np.asarray(features,dtype=float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.all(np.isfinite(values)):
        raise ValueError("finite two-prefactor features required")
    return np.column_stack([np.ones(len(values)),(values-center)/scale])


def fit_conditional_null(*,candidate,samples,features,sample_ids,
        fit_indices,calibration_indices_by_anchor,configuration_id,nominal_coverage=.95):
    names=tuple(samples);arrays={n:np.asarray(v,dtype=float) for n,v in samples.items()}
    if candidate not in arrays or set(sample_ids)!=set(names) or len(names)<2:
        raise ValueError("complete candidate samples required")
    shape=arrays[candidate].shape
    if len(shape)!=2 or any(v.shape!=shape or not np.all(np.isfinite(v)) for v in arrays.values()):
        raise ValueError("finite aligned residual matrices required")
    ids=tuple(sample_ids[candidate])
    if len(ids)!=shape[0] or len(set(ids))!=len(ids) or any(tuple(sample_ids[n])!=ids for n in names):
        raise ValueError("identical ordered observation identities required")
    values=np.asarray(features,dtype=float)
    if values.shape!=(shape[0],2) or not np.all(np.isfinite(values)):
        raise ValueError("aligned candidate-owned prefactor features required")
    fit=np.asarray(fit_indices,dtype=int)
    groups={n:np.asarray(v,dtype=int) for n,v in calibration_indices_by_anchor.items()}
    all_indices=list(fit)+[i for group in groups.values() for i in group]
    if len(fit)<4 or not groups or len(set(all_indices))!=len(all_indices) or any(i<0 or i>=shape[0] for i in all_indices):
        raise ValueError("disjoint valid fit and anchor calibration blocks required")
    if not 0<nominal_coverage<1 or not configuration_id:raise ValueError("explicit coverage/configuration required")
    center=values[fit].mean(axis=0);scale=values[fit].std(axis=0,ddof=1)
    if np.any(scale<=0):raise ValueError("two varying prefactor features required")
    basis=_basis(values,center,scale)
    if np.linalg.matrix_rank(basis[fit])!=3:raise ValueError("conditional regression design is rank deficient")
    targets={n:arrays[candidate]-arrays[n] for n in names if n!=candidate}
    targets["__absolute__"]=arrays[candidate]
    views={}
    for key,target in targets.items():
        coefficients=np.linalg.lstsq(basis[fit],target[fit],rcond=None)[0]
        innovation=target-basis@coefficients
        metric=noise_metric(center=innovation[fit].mean(axis=0),covariance=np.cov(innovation[fit],rowvar=False,ddof=1))
        views[key]=dict(coefficients=coefficients,noise_metric=metric)
    radii={};cal_scores={}
    for anchor,indices in groups.items():
        if len(indices)<1:raise ValueError("nonempty anchor calibration required")
        scores=[]
        for key,target in targets.items():
            view=views[key]
            white=whiten_noise(target[indices]-basis[indices]@view["coefficients"],view["noise_metric"])
            scores.append(np.sum(white**2,axis=1))
        maximum=np.max(scores,axis=0);rank=ceil((len(indices)+1)*nominal_coverage)
        if rank>len(indices):raise ValueError("insufficient finite anchor calibration budget")
        radii[anchor]=float(np.sort(maximum)[rank-1]);cal_scores[anchor]=maximum
    return dict(candidate_id=candidate,universe=names,configuration_id=configuration_id,
        feature_center=center,feature_scale=scale,views=views,
        squared_radius=max(radii.values()),anchor_squared_radii=radii,anchor_calibration_scores=cal_scores,
        nominal_coverage=nominal_coverage,conditional_mean_model="affine fitted oxygen/steam prefactors",
        calibration_rule="maximum of predeclared per-anchor joint order statistics",
        unknown_continuous_parameter_coverage_established=False,reaction_support="oxygen_and_steam")


def conditional_candidate_set(*,profiles,observation_ids,null_models):
    names=tuple(null_models)
    if not names or set(profiles)!=set(names) or set(observation_ids)!=set(names):raise ValueError("complete profile library required")
    if len(set(observation_ids.values()))!=1 or any(not v for v in observation_ids.values()):raise ValueError("same observation required")
    if len({m["configuration_id"] for m in null_models.values()})!=1:raise ValueError("conditional configurations differ")
    retained=[];scores={}
    for name,model in null_models.items():
        if model["candidate_id"]!=name or set(model["universe"])!=set(names):raise ValueError("conditional identity/universe differs")
        if set(model["views"])!=(set(names)-{name})|{"__absolute__"}:raise ValueError("missing conditional view")
        basis=_basis(np.asarray(profiles[name]["amplitude_multipliers"])[None,:],np.asarray(model["feature_center"]),np.asarray(model["feature_scale"]))
        residual=np.asarray(profiles[name]["profiled_residual"],dtype=float)
        view_scores={}
        for key,view in model["views"].items():
            target=residual if key=="__absolute__" else residual-np.asarray(profiles[key]["profiled_residual"])
            innovation=target-basis@np.asarray(view["coefficients"])
            white=whiten_noise(innovation,view["noise_metric"])[0]
            view_scores[key]=float(white@white)
        radius=float(model["squared_radius"])
        if not np.isfinite(radius) or radius<=0:raise ValueError("finite positive conditional radius required")
        score=max(view_scores.values());inside=score<=radius
        scores[name]=dict(view_scores=view_scores,score=score,squared_radius=radius,inside=inside)
        if inside:retained.append(name)
    return dict(candidate_set=retained,scores=scores,parameter_output=None,parameter_confidence_intervals=None,
        reaction_support_conditioned_present=True,uniform_parameter_recovery=False,
        interpretation="experimental_parameter_conditioned_four_transport_set")
