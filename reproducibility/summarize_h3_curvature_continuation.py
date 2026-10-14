"""Read paired continuation results; no reconstruction or optimization."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/_audits/h3_curvature_continuation"
SOURCE = ROOT / "artifacts/_audits/h3_joint_likelihood_development"
METRICS = ("maximum_parameter_relative_error", "maximum_outer_coefficient_relative_error",
           "maximum_mechanism_action_relative_error")


def passed(row):
    return bool(row["parameter_gate_pass"] and row["outer_coefficient_gate_pass"])


def summarize(audit, prior):
    if not audit["complete"] or not prior["complete"]:
        raise ValueError("completed evidence required")
    rows = audit["records"]
    declared = {(role, seed, variant) for role, seed in (("independent", 42303), ("partial", 42302))
                for variant in ("original_coordinates", "curvature_coordinates")}
    indexed = {(r["role"], r["seed"], r["variant"]): r for r in rows}
    if len(rows) != 4 or set(indexed) != declared:
        raise ValueError("four unique declared fits required")
    initial = {(r["role"], r["seed"]): r for r in prior["records"]}
    cases = []
    for role, seed in (("independent", 42303), ("partial", 42302)):
        start = initial[(role, seed)]
        variants = [indexed[(role, seed, v)] for v in ("original_coordinates", "curvature_coordinates")]
        for r in variants:
            if r["signature"] != audit["signature"] or r["input_packet_sha256"] != start["input_packet_sha256"]:
                raise ValueError("protocol or observation identity mismatch")
            if abs(r["initial_nll"]-start["final_nll"]) > 1e-8:
                raise ValueError("continuation start objective mismatch")
            if r["iterations"] > 100 or r["objective_evaluations"] > 250:
                raise ValueError("declared continuation budget exceeded")
        raw, white = variants
        case = {"role": role, "seed": seed, "initial_quantitative_pass": passed(start),
            "initial_errors": {m: start[m] for m in METRICS}, "variants": {}}
        for r in variants:
            case["variants"][r["variant"]] = {**{m: r[m] for m in METRICS},
                **{k: r[k] for k in ("final_nll", "optimizer_success", "optimizer_message", "iterations",
                    "objective_evaluations", "elapsed_seconds", "projected_scaled_gradient_infinity_norm", "reverted_to_start")},
                "quantitative_pass": passed(r), "action_pass": r["mechanism_action_gate_pass"],
                "positive_diffusion_coefficients": all(c >= 0 for eq in r["coefficients"] for c in eq[:2])}
        case["whitened_minus_original_errors"] = {m: white[m]-raw[m] for m in METRICS}
        case["whitened_minus_original_nll"] = white["final_nll"]-raw["final_nll"]
        cases.append(case)
    return {"complete": True, "cases": cases, "existing_packets": 2, "paired_continuations": 4,
        "new_quantitative_passes_vs_saved_start": {v: sum(not c["initial_quantitative_pass"] and c["variants"][v]["quantitative_pass"] for c in cases)
            for v in ("original_coordinates", "curvature_coordinates")},
        "claim_boundary": "Two selected difficult development cases, continuing previous fits; not independent confirmation or full-algorithm superiority. Curvature construction cost is additional to both continuation budgets; no confidence-interval claim."}


def main():
    path = OUT / "audit.json"
    if not path.exists():
        raise SystemExit("Continuation batch not yet complete; no refits started.")
    paths = [Path(__file__), path, SOURCE / "audit.json"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "comparison.json"
    if destination.exists() and json.loads(destination.read_text())["signature"] == signature:
        print("Completed continuation comparison; no calculations repeated.")
        return
    result = summarize(json.loads(path.read_text()), json.loads((SOURCE / "audit.json").read_text()))
    result.update(signature=signature, source_sha256=hashes)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
