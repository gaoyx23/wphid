"""Same-observation covariance of candidate contrasts; not selection."""
import numpy as np


def paired_residual_contrast(*,first,second,baseline_first,baseline_second):
    a,b=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    x,y=np.asarray(baseline_first,dtype=float),np.asarray(baseline_second,dtype=float)
    if a.ndim!=2 or len(a)<2 or a.shape!=b.shape or x.shape!=(a.shape[1],) or y.shape!=x.shape:
        raise ValueError("paired samples and aligned baselines required")
    if not all(np.all(np.isfinite(v)) for v in (a,b,x,y)):
        raise ValueError("finite paired residuals required")
    difference=a-b
    covariance=np.cov(difference,rowvar=False,ddof=1).reshape((len(x),len(x)))
    separate=np.cov(a,rowvar=False,ddof=1).reshape(covariance.shape)+np.cov(b,rowvar=False,ddof=1).reshape(covariance.shape)
    denominator=float(np.trace(separate))
    return dict(sample_count=len(a),mean_contrast=np.mean(difference,axis=0),
        baseline_contrast=x-y,mean_contrast_shift=np.mean(difference,axis=0)-(x-y),
        paired_contrast_covariance=covariance,
        independent_candidate_assumption_covariance=separate,
        paired_to_independent_variance_trace_ratio=(float(np.trace(covariance))/denominator if denominator>0 else None),
        candidate_acceptance=False,
        claim_boundary="Requires matched draw identities. Conditional known-parameter contrasts, not recovery probability or selection thresholds.")
