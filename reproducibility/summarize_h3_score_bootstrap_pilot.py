"""Summarize eight pilot pairs, explicitly no 95% quantile estimation."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.run_h3_score_bootstrap_pilot import OUT, frozen_protocol
from experiments.run_h3_moment_precision_development import OUT as PARENT, save
from experiments.run_h3_curvature_continuation import OUT as START
from reproducibility.diagnose_h3_likelihood_curvature import CASES


def main():
    frozen_protocol()
    path = OUT / "audit.json"
    if not path.exists():
        raise SystemExit("Full pilot required; no partial confidence budget computed.")
    paths = [Path(__file__), path, PARENT / "audit.json", START / "audit.json"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "summary.json"
    if destination.exists():
        if json.loads(destination.read_text())["signature"] != signature:
            raise RuntimeError("preserve pilot summary")
        print("Completed score pilot summary; no calculations repeated.")
        return
    audit = json.loads(path.read_text())
    if not audit["complete"] or len(audit["records"]) != 8:
        raise RuntimeError("eight pilot pairs required")
    parent = json.loads((PARENT / "audit.json").read_text())
    refs = {(r["role"], r["seed"]): r for r in parent["records"] if r["operator"] == "moment_corrected_degree4"}
    starts = json.loads((START / "audit.json").read_text())
    results = []
    for role, seed in CASES:
        reference = next(r for r in starts["records"] if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
        equation = 1 if role == "independent" else 2
        coordinate = len(reference["parameters"][0])*3+4*equation+3
        original_coordinate = reference["coefficients"][equation][3]/max(abs(refs[(role, seed)]["coefficients"][equation][3]), 1e-12)
        pairs = [r for r in audit["records"] if (r["role"], r["seed"]) == (role, seed)]
        scores = [r["profile_score_statistic_raw"] for r in pairs]
        ratios = [r["fits"]["unrestricted"]["vector"][coordinate]/original_coordinate for r in pairs]
        row = {"role": role, "seed": seed, "count": len(pairs), "valid_pairs": sum(r["valid_local_pair"] for r in pairs),
               "score_statistics_raw": scores, "score_min": min(scores), "score_max": max(scores),
               "free_amplitude_to_plugin_reference_ratios": ratios,
               "free_amplitude_within_25_percent_of_plugin_reference": sum(abs(r-1) <= .25 for r in ratios),
               "uses_original_simulation_truth": False, "empirical_95_percent_budget": None}
        results.append(row)
        print(json.dumps(row, indent=2))
    save(destination, {"signature": signature, "source_sha256": hashes, "complete": True,
        "records": results, "budget_calibrated": False, "new_optimizations": 0,
        "claim_boundary": "Eight conditional plug-in bootstrap pairs, four per selected difficult case. Amplitude variation is relative to the bootstrap reference, not original truth. No tail quantile, validated coverage or deployment rule is inferred."})


if __name__ == "__main__":
    main()
