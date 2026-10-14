"""Analyze saved paired fits and evaluate local profiles, without optimizing."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from experiments.run_h3_moment_precision_development import (
    OUT, corrected_problem, groups_from_saved_partition, fingerprint, save, h3)
from wphid.discovery.profile_information import (
    conditional_profile_objective, diagnose_profile_information)


def analyze_one(old, new):
    seed, role = old["seed"], old["role"]
    local, bases, targets = h3._assemble(seed, role)
    nl, nb, nt, identity, _ = corrected_problem(local, seed, role)
    if old["input_packet_sha256"] != identity or new["input_packet_sha256"] != identity:
        raise RuntimeError("reconstructed input packet differs from saved comparison")
    groups = groups_from_saved_partition(old["source_selected_partition"])
    rows = []
    for saved, problem in ((old, (local, bases, targets)), (new, (nl, nb, nt))):
        ll, bb, yy = problem
        builder = h3._builder(ll, h3.FAMILIES[h3.TRUTH_FAMILY])
        params = saved["parameters"][0]
        fitted_atoms = tuple(builder(params[g])[e] for e, g in enumerate(groups))
        # True kinetic vectors are evaluated ONLY as an oracle diagnostic.
        # They never initialize optimization and do not change saved fits.
        oracle_atoms = tuple(builder(h3.TRUTH_CASES[role]["parameters"][e])[e]
                             for e in range(len(h3.FIELDS)))
        fitted = conditional_profile_objective(bb, yy, fitted_atoms)
        oracle = conditional_profile_objective(bb, yy, oracle_atoms)
        rows.append({"operator": saved["operator"], "fitted_profile_objective": fitted,
            "oracle_profile_objective": oracle,
            "fitted_minus_oracle_objective": fitted-oracle,
            "optimizer_success": saved["optimizer_success"],
            "profile_information_at_fitted_parameters": diagnose_profile_information(
                bb, yy, builder, params, groups)})
    return {"seed": seed, "role": role, "input_packet_sha256": identity,
            "new_minus_old_maximum_parameter_error": new["maximum_parameter_relative_error"]-old["maximum_parameter_relative_error"],
            "new_minus_old_maximum_outer_error": new["maximum_outer_coefficient_relative_error"]-old["maximum_outer_coefficient_relative_error"],
            "old_quantitative_pass": old["parameter_gate_pass"] and old["outer_coefficient_gate_pass"],
            "new_quantitative_pass": new["parameter_gate_pass"] and new["outer_coefficient_gate_pass"],
            "operators": rows}


def main():
    source = OUT / "audit.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not payload["complete"] or fingerprint()[0] != payload["signature"]:
        raise RuntimeError("fit audit incomplete or source implementation changed")
    files = (Path(__file__), ROOT / "src/wphid/discovery/profile_information.py", source)
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "analysis.json"
    if destination.exists() and json.loads(destination.read_text())["signature"] == signature:
        print("Completed saved-fit profile analysis; no evaluations repeated.", flush=True)
        return
    pairs = {}
    for row in payload["records"]:
        pairs.setdefault((row["role"], row["seed"]), {})[row["operator"]] = row
    results = []
    for (role, seed), pair in sorted(pairs.items()):
        path = OUT / "analysis_cases" / f"{role}-{seed}.json"
        if path.exists():
            entry = json.loads(path.read_text())
            if entry["signature"] != signature:
                raise RuntimeError("analysis checkpoint signature differs")
        else:
            print(f"{role} {seed}: saved-fit geometry/objective evaluation, no optimization", flush=True)
            entry = analyze_one(pair["legacy_simpson"], pair["moment_corrected_degree4"])
            entry["signature"] = signature
            save(path, entry)
        results.append(entry)
    paired = {"parameter_error_better": sum(r["new_minus_old_maximum_parameter_error"] < 0 for r in results),
              "outer_error_better": sum(r["new_minus_old_maximum_outer_error"] < 0 for r in results),
              "lost_quantitative_pass": sum(r["old_quantitative_pass"] and not r["new_quantitative_pass"] for r in results),
              "gained_quantitative_pass": sum(not r["old_quantitative_pass"] and r["new_quantitative_pass"] for r in results),
              "count": len(results)}
    save(destination, {"signature": signature, "source_sha256": hashes, "complete": True,
        "paired": paired, "records": results, "new_optimizations": 0,
        "new_noise_realizations": 0,
        "interpretation": "Fitted objective below oracle means the noisy profile criterion favors this estimate over truth, not proof of global optimality. Residualized shape singular values are local diagnostics, not calibrated confidence intervals. Previously used development data only."})
    print(json.dumps(paired, indent=2), flush=True)


if __name__ == "__main__":
    main()
