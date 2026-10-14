"""Local weak-balance uncertainty diagnostics, not calibrated selection.

Observation noise is described as L*z, with Cov(z)=I. Columns of L may be
correlated spatial/channel patterns; they are not independent residual rows.
No random draws, hidden fields, candidate thresholds or probability claims.
"""
import numpy as np


def linearize_balance_noise(*, values, noise_modes, evaluate, mode_labels,
                            epsilon=1.e-3, rank_relative_tolerance=1.e-10):
    values=np.array(values,dtype=float,copy=True)
    modes=np.array(noise_modes,dtype=float,copy=True)
    labels=tuple(mode_labels)
    if values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("finite nonempty observation vector required")
    if modes.ndim!=2 or modes.shape[0]!=len(values) or not modes.shape[1] or not np.all(np.isfinite(modes)):
        raise ValueError("finite observation-aligned noise factor required")
    if len(labels)!=modes.shape[1] or len(set(labels))!=len(labels) or any(not isinstance(v,str) or not v for v in labels):
        raise ValueError("unique explicit noise mode labels required")
    if not np.isfinite(epsilon) or epsilon<=0 or not np.isfinite(rank_relative_tolerance) or not 0<rank_relative_tolerance<1:
        raise ValueError("positive finite step and rank tolerance in (0,1) required")
    def residual(v):
        copy=np.array(v,dtype=float,copy=True);copy.setflags(write=False)
        r=np.asarray(evaluate(copy),dtype=float)
        if r.ndim!=1 or not len(r) or not np.all(np.isfinite(r)):
            raise ValueError("evaluator must return a finite nonempty residual vector")
        return r.copy()
    base=residual(values)
    derivatives=[];stability=[];curvature=[];slope_change=[]
    for j in range(modes.shape[1]):
        direction=modes[:,j]
        responses=[residual(values+step*direction) for step in
                   (epsilon,-epsilon,.5*epsilon,-.5*epsilon,1.,-1.)]
        if any(r.shape!=base.shape for r in responses):
            raise ValueError("weak residual dimensions changed under observation perturbation")
        plus,minus,half_plus,half_minus,full_plus,full_minus=responses
        coarse=(plus-minus)/(2*epsilon)
        fine=(half_plus-half_minus)/epsilon
        scale=max(float(np.linalg.norm(fine)),1.e-30)
        stability.append(float(np.linalg.norm(coarse-fine)/scale))
        curvature.append(float(np.linalg.norm(.5*(full_plus+full_minus)-base)/scale))
        slope_change.append(float(np.linalg.norm(.5*(full_plus-full_minus)-fine)/scale))
        derivatives.append(fine)
    factor=np.stack(derivatives,axis=1)
    covariance=factor@factor.T
    u,singular,_=np.linalg.svd(factor,full_matrices=False)
    rank=int(np.sum(singular>rank_relative_tolerance*singular[0])) if len(singular) and singular[0]>0 else 0
    supported=u[:,:rank].T@base
    outside=base-u[:,:rank]@supported
    distance=float(np.sum((supported/singular[:rank])**2)) if rank else 0.
    return dict(residual=base,noise_response_factor=factor,linearized_covariance=covariance,
        marginal_standard_deviation=np.sqrt(np.maximum(np.diag(covariance),0)),
        effective_noise_rank=rank,residual_count=len(base),noise_mode_labels=labels,
        noise_range_mahalanobis_squared=distance,
        residual_outside_noise_range_norm=float(np.linalg.norm(outside)),
        derivative_step_relative_change=np.asarray(stability),
        unit_mode_curvature_relative_to_linear_response=np.asarray(curvature),
        unit_mode_slope_relative_change=np.asarray(slope_change),
        evaluator_calls=1+6*modes.shape[1],new_random_draws=0,
        first_order_only=True,calibrated_selection=False,
        claim_boundary="Conditional on the supplied noise factor and evaluator. Nullspace residual is not discarded. No coverage, adequacy or candidate acceptance threshold.")
