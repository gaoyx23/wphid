"""Three gas-scale corrections retaining declared physical correlations."""
from math import ceil

import numpy as np

from .empirical_noise import noise_metric, whiten_noise


def grouped_noise_diagnostic(*, changes, reference_covariance, groups,
                             fit_count=64, calibration_count=32,
                             nominal_coverage=.95):
    samples = np.asarray(changes, dtype=float)
    covariance = np.asarray(reference_covariance, dtype=float)
    labels = tuple(groups)
    if (samples.ndim != 2 or not np.all(np.isfinite(samples)) or
            len(labels) != samples.shape[1] or any(not isinstance(v, str) or not v for v in labels)):
        raise ValueError("finite samples with one declared physical group per row required")
    if (not isinstance(fit_count, int) or not isinstance(calibration_count, int)
            or fit_count < 2 or calibration_count < 1
            or len(samples) - fit_count - calibration_count < 2):
        raise ValueError("separate fit/calibration/evaluation blocks required")
    if not np.isfinite(nominal_coverage) or not 0. < nominal_coverage < 1.:
        raise ValueError("nominal coverage in (0,1) required")
    noise_metric(center=np.zeros(samples.shape[1]), covariance=covariance)
    fit = samples[:fit_count]
    # Each gas has one variance gain. Cross-window/cross-gas correlations are
    # inherited from the declared local propagation, not estimated from 64 draws.
    variances = np.var(fit, axis=0, ddof=1)
    ratios = variances / np.diag(covariance)
    gains = {label: float(np.mean(ratios[[v == label for v in labels]]))
             for label in dict.fromkeys(labels)}
    if any(not np.isfinite(v) or v <= 0. for v in gains.values()):
        raise ValueError("positive estimated group variance gains required")
    scales = np.sqrt([gains[label] for label in labels])
    corrected = covariance * scales[:, None] * scales[None, :]
    metric = noise_metric(center=fit.mean(axis=0), covariance=corrected)
    calibration = whiten_noise(samples[fit_count:fit_count + calibration_count], metric)
    evaluation = whiten_noise(samples[fit_count + calibration_count:], metric)
    cal_scores = np.sum(calibration**2, axis=1)
    scores = np.sum(evaluation**2, axis=1)
    rank = ceil((calibration_count + 1)*nominal_coverage)
    threshold = float(np.sort(cal_scores)[rank - 1]) if rank <= calibration_count else float("inf")
    white_cov = np.cov(evaluation, rowvar=False, ddof=1).reshape(covariance.shape)
    return dict(fit_count=fit_count, calibration_count=calibration_count,
        evaluation_count=len(evaluation), nominal_coverage=nominal_coverage,
        calibration_order_statistic_rank=rank, group_variance_gains=gains,
        estimated_covariance_parameter_count=len(gains),
        estimated_mean_parameter_count=samples.shape[1],
        reference_correlations_preserved=True, metric=metric,
        calibration_scores=cal_scores, evaluation_scores=scores,
        calibrated_squared_radius=(threshold if np.isfinite(threshold) else None),
        infinite_radius=not np.isfinite(threshold),
        evaluation_inside_count=int(np.sum(scores <= threshold)),
        evaluation_whitened_mean_norm=float(np.linalg.norm(evaluation.mean(axis=0))),
        evaluation_whitened_variance_trace_ratio=float(np.trace(white_cov)/samples.shape[1]),
        evaluation_covariance_relative_frobenius_error=float(np.linalg.norm(white_cov-np.eye(samples.shape[1]))/np.sqrt(samples.shape[1])),
        candidate_selection=False, independent_confirmation=False)
