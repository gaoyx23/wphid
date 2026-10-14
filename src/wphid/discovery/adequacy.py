"""Post-discovery model-adequacy diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def maximum_reference_bound(reference_scores, *, relative_margin=0.10):
    """Return a conservative bound above every finite calibration score."""

    scores = np.asarray(reference_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("reference_scores must be a non-empty finite vector")
    if relative_margin < 0.0 or not np.isfinite(relative_margin):
        raise ValueError("relative_margin must be finite and non-negative")
    return float((1.0 + relative_margin) * np.max(scores))


def one_sided_prediction_upper_bound(reference_scores, *, confidence=0.95):
    """Student-t upper prediction bound for one future scalar score."""

    scores = np.asarray(reference_scores, dtype=float)
    if scores.ndim != 1 or scores.size < 2 or not np.all(np.isfinite(scores)):
        raise ValueError("at least two finite reference scores are required")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and 1")
    critical = float(student_t.ppf(confidence, scores.size - 1))
    upper = float(
        np.mean(scores)
        + critical
        * np.std(scores, ddof=1)
        * np.sqrt(1.0 + 1.0 / scores.size)
    )
    return {
        "upper_bound": upper,
        "critical_value": critical,
        "reference_mean": float(np.mean(scores)),
        "reference_standard_deviation": float(np.std(scores, ddof=1)),
        "reference_count": int(scores.size),
        "confidence": float(confidence),
    }


def absolute_adequacy_decision(score, upper_bound):
    """Accept a library interpretation only when its score is calibrated."""

    if not np.isfinite(score) or not np.isfinite(upper_bound):
        raise ValueError("score and upper_bound must be finite")
    if upper_bound < 0.0:
        raise ValueError("upper_bound must be non-negative")
    accepted = bool(score <= upper_bound)
    return {
        "score": float(score),
        "upper_bound": float(upper_bound),
        "adequacy_pass": accepted,
        "reported_scope": "within_library" if accepted else "out_of_library",
    }


def grouped_probe_diagnostics(base_matrices, targets, selected_columns,
                              probe_columns, condition_count, *,
                              minimum_bic_gain=2.0):
    """Test a non-admissible discrepancy probe across held-out condition groups."""
    if condition_count < 2:
        raise ValueError("at least two condition groups are required")
    diagnostics = []
    for equation, (base, target, selected, probe) in enumerate(zip(
            base_matrices, targets, selected_columns, probe_columns)):
        base = np.asarray(base, dtype=float)
        target = np.asarray(target, dtype=float)
        selected = np.asarray(selected, dtype=float)
        probe = np.asarray(probe, dtype=float)
        if selected.ndim != 2 or selected.shape[0] != target.size:
            raise ValueError("selected columns must be two-dimensional and row-aligned")
        if probe.shape != target.shape or base.shape[0] != target.size:
            raise ValueError("probe, base, and target rows must align")
        if target.size % condition_count:
            raise ValueError("condition groups must have equal row counts")
        row_count = target.size // condition_count
        gains, signs = [], []
        for held_out in range(condition_count):
            rows = np.concatenate([
                np.arange(index * row_count, (index + 1) * row_count)
                for index in range(condition_count) if index != held_out
            ])
            reduced = np.column_stack((base[rows], selected[rows]))
            full = np.column_stack((reduced, probe[rows]))
            reduced_coef = np.linalg.lstsq(reduced, target[rows], rcond=None)[0]
            full_coef = np.linalg.lstsq(full, target[rows], rcond=None)[0]
            reduced_residual = target[rows] - reduced @ reduced_coef
            full_residual = target[rows] - full @ full_coef
            reduced_rss = max(float(reduced_residual @ reduced_residual),
                              np.finfo(float).tiny)
            full_rss = max(float(full_residual @ full_residual), np.finfo(float).tiny)
            gains.append(float(len(rows) * np.log(reduced_rss / full_rss)
                               - np.log(len(rows))))
            signs.append(int(np.sign(full_coef[-1])))
        evidence_rate = float(np.mean([
            gain >= minimum_bic_gain for gain in gains
        ]))
        sign_consistency = float(max(signs.count(-1), signs.count(1)) / len(signs))
        diagnostics.append({
            "equation": equation, "bic_gains": gains, "signs": signs,
            "evidence_rate": evidence_rate,
            "sign_consistency": sign_consistency,
        })
    return diagnostics
