"""Finite-profile resolution witnesses, without a probability threshold.

For candidate amplitude a, a point estimate satisfying relative tolerance e
must lie in [(1-e)a,(1+e)a] (endpoints sorted for negative amplitudes).
Disjoint intervals prove only a conditional incompatibility among sampled
explanations; they do not prove both explanations are true/plausible.
"""
import itertools
import numpy as np


def resolution_witness(points, relative_tolerance):
    if not 0 < relative_tolerance < 1 or len(points) < 2:
        raise ValueError("two or more points and a relative tolerance in (0,1) required")
    normalized = []
    for point in points:
        amplitude, cost = float(point["amplitude"]), float(point["score_cost"])
        if amplitude == 0 or not np.isfinite(amplitude) or not np.isfinite(cost) or cost < -1e-8:
            raise ValueError("finite nonzero amplitudes and nonnegative relative costs required")
        lower, upper = sorted(((1-relative_tolerance)*amplitude, (1+relative_tolerance)*amplitude))
        normalized.append({**point, "admissible_point_estimate_interval": [lower, upper]})
    witnesses = []
    for first, second in itertools.combinations(normalized, 2):
        lower = max(first["admissible_point_estimate_interval"][0], second["admissible_point_estimate_interval"][0])
        upper = min(first["admissible_point_estimate_interval"][1], second["admissible_point_estimate_interval"][1])
        if lower > upper+1e-12*max(abs(lower), abs(upper), 1.):
            witnesses.append({"first": first, "second": second,
                "required_score_budget": max(first["score_cost"], second["score_cost"]),
                "nonoverlap_gap": lower-upper})
    best = min(witnesses, key=lambda row: row["required_score_budget"]) if witnesses else None
    return {"relative_tolerance": relative_tolerance, "sampled_points": normalized,
            "minimum_sampled_score_budget_for_incompatible_point_targets": None if best is None else best["required_score_budget"],
            "witness": best,
            "qualification": "If both witness explanations are retained, no one amplitude estimate meets the declared relative tolerance against both. No calibrated confidence budget, population guarantee, truth selection or global profile boundary is supplied."}
