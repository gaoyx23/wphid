"""Compare completed precision developments from saved values, never refit.

The unit is an existing observation packet, not an optimizer evaluation.
This is an internal development comparison, not independent confirmation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/_audits/h3_joint_likelihood_development"
METRICS = (
    "maximum_parameter_relative_error",
    "maximum_outer_coefficient_relative_error",
    "maximum_mechanism_action_relative_error",
)


def quantitative_pass(row):
    return bool(row["parameter_gate_pass"] and row["outer_coefficient_gate_pass"])


def median(values):
    values = sorted(values)
    n = len(values)
    return (values[(n-1)//2] + values[n//2]) / 2


def compare_saved(moment, covariance, likelihood):
    """Reject incomplete/mismatched evidence rather than silently drop cases."""
    for audit in (moment, covariance, likelihood):
        if not audit.get("complete"):
            raise ValueError("only completed audits may be compared")
    estimators = {}
    for name, audit, operator in (
        ("legacy_simpson", moment, "legacy_simpson"),
        ("moment_corrected", moment, "moment_corrected_degree4"),
        ("fixed_covariance", covariance, None),
        ("joint_parameter_dependent_likelihood", likelihood, None),
    ):
        rows = [r for r in audit["records"] if operator is None or r["operator"] == operator]
        indexed = {(r["role"], r["seed"]): r for r in rows}
        expected = {(role, seed) for role in ("partial", "independent") for seed in range(42301, 42305)}
        if len(rows) != 8 or set(indexed) != expected:
            raise ValueError(f"{name}: eight unique declared packets required")
        if any(r["signature"] != audit["signature"] for r in rows):
            raise ValueError(f"{name}: checkpoint signature mismatch")
        estimators[name] = indexed
    keys = sorted(estimators["moment_corrected"])
    for key in keys:
        if len({rows[key]["input_packet_sha256"] for rows in estimators.values()}) != 1:
            raise ValueError(f"observation packet mismatch: {key}")
    summaries = []
    for name, rows in estimators.items():
        for role in ("partial", "independent"):
            selected = [r for (rr, _), r in rows.items() if rr == role]
            summaries.append({"estimator": name, "role": role, "count": len(selected),
                "quantitative_pass": sum(map(quantitative_pass, selected)),
                "action_pass": sum(bool(r["mechanism_action_gate_pass"]) for r in selected),
                **{f"median_{metric}": median([r[metric] for r in selected]) for metric in METRICS}})
    paired = []
    baseline = estimators["moment_corrected"]
    for name in ("fixed_covariance", "joint_parameter_dependent_likelihood"):
        rows = estimators[name]
        paired.append({"estimator": name, "reference": "moment_corrected", "count": 8,
            "quantitative_pass_gained": sum(not quantitative_pass(baseline[k]) and quantitative_pass(rows[k]) for k in keys),
            "quantitative_pass_lost": sum(quantitative_pass(baseline[k]) and not quantitative_pass(rows[k]) for k in keys),
            **{f"improved_{metric}": sum(rows[k][metric] < baseline[k][metric] for k in keys) for metric in METRICS}})
    cases = []
    for role, seed in keys:
        item = {"role": role, "seed": seed, "estimators": {}}
        for name, rows in estimators.items():
            r = rows[(role, seed)]
            item["estimators"][name] = {**{m: r[m] for m in METRICS},
                "quantitative_pass": quantitative_pass(r), "action_pass": bool(r["mechanism_action_gate_pass"])}
        r = estimators["joint_parameter_dependent_likelihood"][(role, seed)]
        item["likelihood_optimizer"] = {k: r[k] for k in (
            "optimizer_success", "optimizer_message", "iterations", "function_evaluations",
            "initial_nll", "final_nll", "oracle_nll_post_fit_only", "elapsed_seconds")}
        item["likelihood_minus_moment_errors"] = {m: r[m] - baseline[(role, seed)][m] for m in METRICS}
        cases.append(item)
    return {"complete": True, "summary": summaries, "paired": paired, "cases": cases,
        "likelihood_optimizer_converged": sum(r["optimizer_success"] for r in estimators["joint_parameter_dependent_likelihood"].values()),
        "existing_observation_packets": 8, "new_noise_realizations": 0, "new_structure_searches": 0,
        "conditional_fits_across_three_developments": 32,
        "claim_boundary": "Internal, fixed-structure development; likelihood adds calibrated noise input and changes optimizer and objective jointly. Not an independent benchmark or isolated ablation. Non-converged fits are retained; these results do not prove global estimator quality."}


def main():
    paths = [ROOT / "artifacts/_audits/h3_moment_precision_development/audit.json",
             ROOT / "artifacts/_audits/h3_covariance_precision_development/audit.json", OUT / "audit.json"]
    if not all(p.exists() for p in paths):
        raise SystemExit("Likelihood batch is incomplete; no partial comparison written and no refits started.")
    audits = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    result = compare_saved(*audits)
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *paths]}
    result["source_sha256"] = hashes
    result["signature"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "comparison.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8"))["signature"] == result["signature"]:
        print("Completed saved-value comparison; no scientific computation repeated.")
        return
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"paired": result["paired"], "likelihood_optimizer_converged": result["likelihood_optimizer_converged"]}, indent=2))


if __name__ == "__main__":
    main()
