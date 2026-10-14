"""Split finite-amplitude residual-noise diagnostics, not mechanism selection."""
from math import ceil

import numpy as np


def noise_metric(*, center, covariance, eigenvalue_floor=1.e-6):
    """Scale units before explicit correlation-eigenvalue regularization."""
    mean = np.asarray(center, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if mean.ndim != 1 or not mean.size or cov.shape != (mean.size, mean.size):
        raise ValueError("aligned center and covariance required")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
        raise ValueError("finite inputs required")
    if not np.allclose(cov, cov.T, rtol=1.e-12, atol=0.):
        raise ValueError("symmetric covariance required")
    if not np.isfinite(eigenvalue_floor) or not 0. < eigenvalue_floor < 1.:
        raise ValueError("explicit correlation eigenvalue floor must be in (0,1)")
    if np.any(np.diag(cov) <= 0.):
        raise ValueError("deterministic noise coordinates need separate treatment")
    scale = np.sqrt(np.diag(cov))
    correlation = cov / scale[:, None] / scale[None, :]
    values, vectors = np.linalg.eigh(correlation)
    if values.min() < -1.e-10:
        raise ValueError("positive semidefinite covariance required")
    regularized = np.maximum(values, eigenvalue_floor)
    # Column samples are transformed to rows with an identity reference covariance.
    transform = vectors / np.sqrt(regularized)[None, :]
    return dict(center=mean, scale=scale, transform=transform,
                correlation_eigenvalues=values,
                eigenvalue_floor=eigenvalue_floor,
                floored_eigenvalue_count=int(np.sum(values < eigenvalue_floor)))


def whiten_noise(samples, metric):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(metric["center"]):
        raise ValueError("aligned sample matrix required")
    if not np.all(np.isfinite(values)):
        raise ValueError("finite samples required")
    return ((values - metric["center"]) / metric["scale"]) @ metric["transform"]


def split_noise_diagnostic(*, changes, predicted_covariance,
                           fit_count=64, calibration_count=32,
                           nominal_coverage=.95, eigenvalue_floor=1.e-6):
    """Fit only first block; rank-calibrate second; diagnose untouched third.

    All inputs are r(noisy observation)-r(clean observation), NOT candidate
    residuals to test against zero. The clean baseline is oracle development
    information. Sample splitting cannot undo earlier exploratory use of data.
    """
    samples = np.asarray(changes, dtype=float)
    if samples.ndim != 2 or not np.all(np.isfinite(samples)):
        raise ValueError("finite sample matrix required")
    if (not isinstance(fit_count, int) or not isinstance(calibration_count, int)
            or fit_count < 2 or calibration_count < 1
            or len(samples) - fit_count - calibration_count < 2):
        raise ValueError("nonempty separate fit/calibration/evaluation blocks required")
    if not np.isfinite(nominal_coverage) or not 0. < nominal_coverage < 1.:
        raise ValueError("nominal coverage must be in (0,1)")
    fit = samples[:fit_count]
    calibration = samples[fit_count:fit_count + calibration_count]
    evaluation = samples[fit_count + calibration_count:]
    empirical = np.cov(fit, rowvar=False, ddof=1).reshape((samples.shape[1],) * 2)
    metrics = {
        "first_order_zero_mean": noise_metric(center=np.zeros(samples.shape[1]),
            covariance=predicted_covariance, eigenvalue_floor=eigenvalue_floor),
        "finite_amplitude_empirical": noise_metric(center=fit.mean(axis=0),
            covariance=empirical, eigenvalue_floor=eigenvalue_floor),
    }
    rank = ceil((calibration_count + 1) * nominal_coverage)
    output = {}
    for name, metric in metrics.items():
        cal_white = whiten_noise(calibration, metric)
        test_white = whiten_noise(evaluation, metric)
        cal_scores = np.sum(cal_white ** 2, axis=1)
        test_scores = np.sum(test_white ** 2, axis=1)
        # Finite order statistic; never use evaluation to pick the threshold.
        threshold = float(np.sort(cal_scores)[rank - 1]) if rank <= calibration_count else float("inf")
        white_cov = np.cov(test_white, rowvar=False, ddof=1).reshape(empirical.shape)
        output[name] = dict(metric=metric, calibration_scores=cal_scores,
            evaluation_scores=test_scores, calibrated_squared_radius=(threshold if np.isfinite(threshold) else None),
            infinite_radius=not np.isfinite(threshold),
            evaluation_inside_count=int(np.count_nonzero(test_scores <= threshold)),
            evaluation_count=len(evaluation),
            evaluation_whitened_mean_norm=float(np.linalg.norm(test_white.mean(axis=0))),
            evaluation_whitened_variance_trace_ratio=float(np.trace(white_cov)/samples.shape[1]),
            evaluation_covariance_relative_frobenius_error=float(np.linalg.norm(white_cov-np.eye(samples.shape[1]))/np.sqrt(samples.shape[1])))
    return dict(fit_count=fit_count, calibration_count=calibration_count,
        evaluation_count=len(evaluation), nominal_coverage=nominal_coverage,
        calibration_order_statistic_rank=rank, methods=output,
        candidate_selection=False, independent_scientific_confirmation=False,
        claim_boundary="Noise changes around a known clean baseline, fixed source and parameters. Reused development data; no mechanism coverage or posterior weights.")
