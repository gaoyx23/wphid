"""Paired structural contrasts with jointly calibrated absolute guard views.

This restricted fixed-parameter prototype is not a general weak selector.
Contrasts cancel common residual shifts, so contrasts alone cannot establish
model adequacy. Every score uses the same observation across candidates.
"""
from math import ceil
import numpy as np
from .empirical_noise import noise_metric, whiten_noise


def absolute_score(residuals, model):
    metric = dict(model["noise_metric"])
    metric["center"] = np.asarray(metric["center"]) + np.asarray(model["candidate_simulated_clean_baseline"])
    return np.sum(whiten_noise(residuals, metric)**2, axis=1)


def fit_paired_null(*, candidate, samples, sample_ids, absolute_models,
                    observation_configuration_id, fit_count=64,
                    calibration_count=32, nominal_coverage=.95):
    names = tuple(samples)
    arrays = {name:np.asarray(values, dtype=float) for name, values in samples.items()}
    if candidate not in arrays or len(names)<2 or set(sample_ids)!=set(names):
        raise ValueError("full candidate samples with matched observation identities required")
    shape = arrays[candidate].shape
    if len(shape)!=2 or any(v.shape!=shape or not np.all(np.isfinite(v)) for v in arrays.values()):
        raise ValueError("finite aligned candidate sample matrices required")
    ids = tuple(sample_ids[candidate])
    if len(ids)!=shape[0] or len(set(ids))!=len(ids) or any(tuple(sample_ids[n])!=ids for n in names):
        raise ValueError("candidate draws must have identical ordered unique identities")
    if (not isinstance(fit_count,int) or not isinstance(calibration_count,int)
            or fit_count<2 or calibration_count<1 or fit_count+calibration_count>shape[0]):
        raise ValueError("separate fit and calibration blocks required")
    if not 0.<nominal_coverage<1. or not observation_configuration_id:
        raise ValueError("coverage and observation configuration required")
    cut = slice(fit_count, fit_count+calibration_count)
    contrasts, calibration_views = {}, []
    for other in names:
        if other==candidate: continue
        difference = arrays[candidate]-arrays[other]
        fit = difference[:fit_count]
        metric = noise_metric(center=fit.mean(axis=0),
            covariance=np.cov(fit,rowvar=False,ddof=1).reshape((shape[1],)*2))
        contrasts[other] = metric
        calibration_views.append(np.sum(whiten_noise(difference[cut],metric)**2,axis=1))
    paired_calibration = np.max(calibration_views,axis=0)
    rank = ceil((calibration_count+1)*nominal_coverage)
    def radius(scores):
        return float(np.sort(scores)[rank-1]) if rank<=calibration_count else None
    joint = {}
    for method, model in absolute_models.items():
        if model["candidate_id"]!=candidate or model["observation_configuration_id"]!=observation_configuration_id:
            raise ValueError("absolute null identity/configuration mismatch")
        scores = np.maximum(paired_calibration,absolute_score(arrays[candidate][cut],model))
        joint[method] = dict(absolute_model=model, calibration_scores=scores,
            calibrated_squared_radius=radius(scores))
    return dict(candidate_id=candidate, universe=names,
        observation_configuration_id=observation_configuration_id,
        contrast_metrics=contrasts, paired_calibration_scores=paired_calibration,
        paired_squared_radius=radius(paired_calibration), joint_views=joint,
        fit_count=fit_count, calibration_count=calibration_count,
        calibration_order_statistic_rank=rank, nominal_coverage=nominal_coverage,
        contrast_baselines_from_candidate_simulations_only=True)


def paired_candidate_set(*, residuals, observation_ids, null_models, absolute_method=None):
    names = tuple(null_models)
    if not names or set(residuals)!=set(names) or set(observation_ids)!=set(names):
        raise ValueError("complete declared library and observation identities required")
    ids = set(observation_ids.values())
    if len(ids)!=1 or None in ids or "" in ids:
        raise ValueError("all candidate residuals must share one observation")
    configs = {m["observation_configuration_id"] for m in null_models.values()}
    if len(configs)!=1: raise ValueError("null configurations differ")
    retained, details = [], {}
    for name, model in null_models.items():
        if model["candidate_id"]!=name or set(model["universe"])!=set(names):
            raise ValueError("candidate null identity or universe differs")
        if set(model["contrast_metrics"])!=set(names)-{name}:
            raise ValueError("missing declared pairwise contrast")
        contrast_scores = {}
        for other,metric in model["contrast_metrics"].items():
            difference = np.asarray(residuals[name],dtype=float)-np.asarray(residuals[other],dtype=float)
            white = whiten_noise(difference[None,:],metric)[0]
            contrast_scores[other] = float(white@white)
        score = max(contrast_scores.values())
        absolute = None
        radius = model["paired_squared_radius"]
        if absolute_method is not None:
            if absolute_method not in model["joint_views"]: raise ValueError("uncalibrated absolute view")
            view = model["joint_views"][absolute_method]
            absolute = float(absolute_score(np.asarray(residuals[name])[None,:],view["absolute_model"])[0])
            score = max(score,absolute)
            radius = view["calibrated_squared_radius"]
        if radius is None or not np.isfinite(radius) or radius<=0.:
            raise ValueError("finite calibrated joint radius required")
        inside = score<=radius
        details[name] = dict(contrast_scores=contrast_scores,absolute_score=absolute,
            score=score,squared_radius=radius,inside=inside)
        if inside: retained.append(name)
    return dict(candidate_set=retained,scores=details,
        absolute_view_included=absolute_method is not None,
        engineering_confirmation=False,unknown_parameter_recovery=False,
        interpretation="relative_contrasts_only" if absolute_method is None else "conditional_closed_library_joint_views")
