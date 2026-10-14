"""Audit saved H3-V6 outcomes; never simulate, optimize, or change decisions."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/paper_h3_v6_identifiability_tier/metrics.json"
OUTPUT = ROOT / "docs/research/algorithm-evidence/joint-recovery-bottlenecks.json"
POSITIVE_ROLES = ("shared", "partial", "independent")


def summarize(records):
    groups = []
    details = []
    for split in ("development", "confirmation"):
        for role in POSITIVE_ROLES:
            rows = [r for r in records if r["split"] == split and r["role"] == role]
            counts = Counter()
            bad_edges = Counter()
            for row in rows:
                structural = bool(row["endpoint_success"])
                quantitative = bool(structural and row["parameter_gate_pass"]
                                    and row["outer_coefficient_gate_pass"])
                full = row["output_tier"] == "full_decomposition"
                counts["truth_screen_admitted"] += bool(row["truth_screen_admitted"])
                counts["structural_endpoint_correct"] += structural
                counts["inner_parameter_pass"] += bool(row["parameter_gate_pass"])
                counts["outer_coefficient_pass"] += bool(row["outer_coefficient_gate_pass"])
                counts["pooled_quantitative_pass"] += quantitative
                counts["full_output"] += full
                counts["quantitative_pass_withheld"] += quantitative and not full
                counts["inaccurate_full_output"] += full and not quantitative
                counts["estimation_failure"] += structural and not quantitative
                errors = row["outer_coefficient_relative_errors"]
                worst = max(errors, key=errors.get) if errors else None
                if structural and not quantitative and worst is not None:
                    bad_edges[worst] += 1
                details.append({
                    "split": split, "role": role, "seed": row["seed"],
                    "source_record_index": records.index(row),
                    "structural_endpoint_correct": structural,
                    "pooled_quantitative_pass": quantitative,
                    "output_tier": row["output_tier"],
                    "worst_outer_coefficient_edge": worst,
                    "maximum_parameter_relative_error": row["maximum_parameter_relative_error"],
                    "maximum_outer_coefficient_relative_error": row["maximum_outer_coefficient_relative_error"],
                    "single_condition_amplitude_diagnostics": row["amplitude_stability"],
                    "pooled_leave_one_condition_out_edge_evidence": row["edge_evidence"],
                })
            groups.append({"split": split, "role": role, "count": len(rows),
                           **dict(counts), "worst_outer_edge_among_estimation_failures": dict(bad_edges)})
    return {"groups": groups, "details": details,
            "interpretation": {
                "truth_used_for": "post-hoc scoring only, never an estimator or decision rule",
                "confirmation_reuse": "Previously inspected confirmation records are diagnostic evidence, not fresh validation for a new method.",
                "withheld": "Numerically accurate pooled estimates withheld by the old reporting rule; not proof that removing that rule is safe.",
                "unresolved": "Saved V6 records omit fitted inner vectors, design matrices and optimizer traces. Conditioning, local minima and errors-in-variables are not distinguished by this audit.",
                "scope": "One reaction mechanism with open family/topology/parameter partition, not simultaneous recovery of all H1 mechanism classes.",
            }}


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    report = summarize(payload["records"])
    report.update(source=str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_protocol_sha256=payload["protocol_sha256"],
                  new_simulations=0, new_noise_draws=0, new_parameter_fits=0,
                  historical_decisions_modified=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
