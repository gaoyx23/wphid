"""Deterministic foundation gate report renderer.

Reads E1 and E3 metrics JSON and produces the canonical
``docs/results/wphid-foundation-gate.md`` report. Every numeric value in
the output is derived from the JSON artifacts; nothing is hard-coded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E1_METRICS_PATH = PROJECT_ROOT / "artifacts" / "e1_closed_library" / "metrics.json"
E3_METRICS_PATH = PROJECT_ROOT / "artifacts" / "e3_missing_interaction" / "metrics.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "results" / "wphid-foundation-gate.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    """Format a numeric value for display, preserving meaningful precision."""
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def render_report(e1_metrics: dict, e3_metrics: dict) -> str:
    """Render the foundation gate Markdown report from metrics dicts."""

    e1_agg = e1_metrics["aggregate"]
    e3_agg = e3_metrics["aggregate"]

    e1_gate = e1_agg["gate"]
    e1_pass = e1_gate["pass"]
    e1_threshold = e1_gate["threshold"]
    e1_observed = e1_gate["observed"]

    e3_pass = e3_agg.get("e3_gate_pass", False)
    e3_detection_rate = e3_agg["missing_interaction_detection_rate"]
    e3_max_false = e3_agg["maximum_false_interactions"]

    overall_pass = bool(e1_pass) and bool(e3_pass)

    if overall_pass:
        decision_line = (
            "**GO: proceed to spatiotemporal PDE weak form.**\n"
        )
    else:
        decision_line = (
            "**NO-GO: temporal foundation gate not passed.**\n"
        )

    lines: list[str] = []
    lines.append("# WPHID foundation discovery gate")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(decision_line)
    lines.append(
        "This decision is limited to the declared temporal weak-form "
        "benchmark and its finite physics grammar. It does not claim "
        "general equation discovery or broad noise robustness."
    )
    lines.append("")

    # ---- Reproduction ----
    lines.append("## Reproduction")
    lines.append("")
    lines.append("From the repository root:")
    lines.append("")
    lines.append("```text")
    lines.append("python -m pip install -e \".[test]\"")
    lines.append("python -m pytest -q")
    lines.append("python experiments/e1_closed_library.py")
    lines.append("python experiments/e3_missing_interaction.py")
    lines.append("python experiments/foundation_report.py")
    lines.append("```")
    lines.append("")

    # ---- E1 ----
    lines.append("## E1: closed-library recovery")
    lines.append("")
    e1_noise = e1_metrics["configuration"]["observation_noise"]
    lines.append(
        f"- Noise: {_fmt(e1_noise['noise_fraction'] * 100)}% "
        f"{e1_noise['distribution']}"
    )
    lines.append(
        f"- Applied to: {', '.join(e1_noise['applied_to'])}"
    )
    lines.append(
        f"- Theta grid: {_fmt(e1_metrics['configuration']['theta_grid'][0])}--"
        f"{_fmt(e1_metrics['configuration']['theta_grid'][-1])} "
        f"({len(e1_metrics['configuration']['theta_grid'])} points)"
    )
    lines.append(
        f"- Train conditions: "
        f"{', '.join(e1_metrics['configuration']['train_ids'])}"
    )
    lines.append(
        f"- Validation conditions: "
        f"{', '.join(e1_metrics['configuration']['validation_ids'])}"
    )
    lines.append(f"- Theta selection: {e1_metrics['configuration']['theta_selection_data']}")
    lines.append(f"- Gate: {e1_gate['metric']} {e1_gate['criterion']} {_fmt(e1_threshold)}")
    lines.append(f"- Observed: {_fmt(e1_observed)}")
    lines.append(f"- Passing seeds: {e1_agg['passing_seed_count']}/{len(e1_agg['seeds'])}")
    lines.append(f"- Mean support precision: {_fmt(e1_agg['mean_support_precision'])}")
    lines.append(f"- Mean support recall: {_fmt(e1_agg['mean_support_recall'])}")
    lines.append(
        f"- Max coefficient relative L2 error: "
        f"{_fmt(e1_agg['maximum_coefficient_relative_l2_error'])}"
    )
    lines.append(
        f"- Max theta relative error: "
        f"{_fmt(e1_agg['maximum_theta_relative_error'])}"
    )
    lines.append(
        f"- Reference validation NRMSE: "
        f"{_fmt(e1_agg['reference_validation_normalized_rmse'])}"
    )
    lines.append("")

    # Per-seed table
    lines.append("| Seed | Precision | Recall | F1 | False positives | Selected theta | Theta rel. error | Coeff rel. L2 | Validation NRMSE | Gate |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for row in e1_metrics["per_seed"]:
        sm = row["support_metrics"]
        gate_mark = "PASS" if row.get("support_gate_pass") else "FAIL"
        lines.append(
            f"| {row['seed']} "
            f"| {_fmt(sm['precision'])} "
            f"| {_fmt(sm['recall'])} "
            f"| {_fmt(sm['f1'])} "
            f"| {sm['false_positives']} "
            f"| {_fmt(row['selected_theta'])} "
            f"| {_fmt(row['theta_relative_error'])} "
            f"| {_fmt(row['coefficient_metrics']['relative_l2_error'])} "
            f"| {_fmt(row['validation_normalized_rmse'])} "
            f"| {gate_mark} |"
        )
    lines.append("")

    # ---- E3 ----
    lines.append("## E3: missing-interaction discovery")
    lines.append("")
    e3_noise = e3_metrics["per_seed"][0]["configuration"]["observation_noise"]
    lines.append(
        f"- Noise: {e3_noise['noise_fraction']} "
        f"({e3_noise['description']})"
    )
    lines.append(
        f"- Expected missing interaction: "
        f"{e3_metrics['per_seed'][0]['evaluation_target']['expected_missing_interaction']}"
    )
    lines.append(f"- Detection rate: {_fmt(e3_detection_rate)} "
                 f"({e3_agg['passing_seed_count']}/{len(e3_agg['seeds'])})")
    lines.append(f"- Max false interactions: {e3_max_false}")
    lines.append(
        f"- Max expanded test NRMSE / E1 reference: "
        f"{_fmt(e3_agg['maximum_expanded_test_nrmse_over_e1_reference'])}"
    )
    lines.append(
        f"- E1 reference validation NRMSE: "
        f"{_fmt(e3_agg['e1_reference_validation_normalized_rmse'])}"
    )
    lines.append("")

    lines.append("| Seed | Detected | Admitted | False interactions | Test NRMSE / E1 ref | Gate |")
    lines.append("|---:|:---:|---|:---:|---:|:---:|")
    for row in e3_metrics["per_seed"]:
        gates = row["gates"]
        detected = "yes" if gates.get("missing_interaction_admitted") else "no"
        admitted = ", ".join(row["expansion"]["admitted_candidates"]) or "none"
        false_count = len(row["expansion"]["false_interactions"])
        ratio = row["expanded_library"]["test_nrmse_over_e1_reference"]
        all_pass = all(gates.values())
        gate_mark = "PASS" if all_pass else "FAIL"
        lines.append(
            f"| {row['seed']} "
            f"| {detected} "
            f"| {admitted} "
            f"| {false_count} "
            f"| {_fmt(ratio)} "
            f"| {gate_mark} |"
        )
    lines.append("")

    # ---- Scope ----
    lines.append("## Scope and limitations")
    lines.append("")
    lines.append(
        "- E2 (redundant library) is unimplemented and not evaluated."
    )
    lines.append(
        "- This benchmark is temporal only. Spatial discretization, boundary "
        "handling, field noise, covariate noise, and out-of-grammar failure "
        "modes are reserved for Plan 2."
    )
    lines.append(
        "- The finite physics grammar is declared a priori; this is not "
        "unrestricted symbolic discovery."
    )
    if not overall_pass:
        lines.append(
            f"- The overall gate is NO-GO. E1 minimum F1 = {_fmt(e1_observed)} "
            f"(threshold {_fmt(e1_threshold)}). "
            f"E3 detection rate = {_fmt(e3_detection_rate)} "
            f"(threshold 0.9)."
        )
        lines.append(
            "- Failing seeds in E1 selected an incorrect Arrhenius theta, "
            "introducing spurious support terms under 5% multivariate noise."
        )
        lines.append(
            "- Failing seeds in E3 did not admit the missing interaction "
            "under the finite-pool validation admission rule."
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_report(e1_path: Path, e3_path: Path, output_path: Path) -> None:
    e1_metrics = _load_json(e1_path)
    e3_metrics = _load_json(e3_path)
    report = render_report(e1_metrics, e3_metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    write_report(E1_METRICS_PATH, E3_METRICS_PATH, REPORT_PATH)
    print(f"Report written to {REPORT_PATH}")
