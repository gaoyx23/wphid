"""Finite-noise moment checks; no candidate selection or coverage guarantee."""
import numpy as np


def compare_noise_moments(*,residual_samples,baseline,predicted_covariance):
    samples=np.asarray(residual_samples,dtype=float)
    baseline=np.asarray(baseline,dtype=float)
    covariance=np.asarray(predicted_covariance,dtype=float)
    if samples.ndim!=2 or len(samples)<2 or baseline.shape!=(samples.shape[1],) or covariance.shape!=(samples.shape[1],samples.shape[1]):
        raise ValueError("aligned samples, baseline and covariance required")
    if not all(np.all(np.isfinite(v)) for v in (samples,baseline,covariance)) or not np.allclose(covariance,covariance.T,rtol=1.e-12,atol=1.e-14):
        raise ValueError("finite inputs and symmetric covariance required")
    try:factor=np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:raise ValueError("positive definite covariance required; no silent jitter") from exc
    change=samples-baseline
    whitened=np.linalg.solve(factor,change.T).T
    empirical=np.cov(change,rowvar=False,ddof=1).reshape(covariance.shape)
    white_covariance=np.cov(whitened,rowvar=False,ddof=1).reshape(covariance.shape)
    ratios=np.diag(empirical)/np.diag(covariance)
    mean=np.mean(whitened,axis=0)
    return dict(sample_count=len(samples),residual_count=len(baseline),
        variance_ratio_min=float(ratios.min()),variance_ratio_median=float(np.median(ratios)),variance_ratio_max=float(ratios.max()),
        whitened_variance_trace_ratio=float(np.trace(white_covariance)/len(baseline)),
        whitened_covariance_relative_frobenius_error=float(np.linalg.norm(white_covariance-np.eye(len(baseline)))/np.sqrt(len(baseline))),
        whitened_mean_change_norm=float(np.linalg.norm(mean)),
        sampling_only_mean_norm_rms_reference=float(np.sqrt(len(baseline)/len(samples))),
        gaussian_sampling_covariance_rms_reference=float(np.sqrt((len(baseline)+1)/(len(samples)-1))),
        empirical_covariance=empirical,mean_change=np.mean(change,axis=0),
        calibrated_acceptance=False)
