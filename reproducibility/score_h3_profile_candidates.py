"""Post-hoc truth scoring of already frozen diagnostic profile points.

Never select a deployed estimate from truth. No optimization or new grid.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from experiments.run_h3_nonlinear_amplitude_profile import OUT, protocol
from experiments.run_h3_joint_likelihood_development import build_model
from experiments.run_h3_moment_precision_development import OUT as PARENT, h3, save
from reproducibility.diagnose_h3_likelihood_curvature import CASES


def main():
    torch.set_num_threads(4)
    protocol()
    paths = [Path(__file__), OUT / "audit.json", PARENT / "audit.json"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "truth-scoring.json"
    if destination.exists():
        if json.loads(destination.read_text())["signature"] != signature:
            raise RuntimeError("preserve post-hoc truth scoring")
        print("Completed post-hoc scoring; no evaluations repeated.")
        return
    audit = json.loads((OUT / "audit.json").read_text())
    if not audit["complete"]:
        raise RuntimeError("all profile points must be frozen before scoring")
    parent = json.loads((PARENT / "audit.json").read_text())
    refs = {(r["role"], r["seed"]): r for r in parent["records"] if r["operator"] == "moment_corrected_degree4"}
    records = []
    for role, seed in CASES:
        model, bases, groups = build_model(refs[(role, seed)], "cpu")
        for row in [r for r in audit["records"] if (r["role"], r["seed"]) == (role, seed)]:
            selected = {"fit": SimpleNamespace(parameters=(tuple(np.asarray(row["parameters"])),),
                coefficients=tuple(np.asarray(row["coefficients"]))), "groups": groups,
                "family": "arrhenius_oxygen_damage", "topology": (True, True, True)}
            errors = h3._quantitative_errors(role, selected, refs[(role, seed)]["source_selected_partition"], bases)
            records.append({"role": role, "seed": seed, "ratio": row["ratio"],
                "unnormalized_nll_difference": row["unnormalized_nll_difference"], **errors})
        del model
    result = {"signature": signature, "source_sha256": hashes, "complete": True, "records": records,
        "optimizations": 0, "truth_used_for_point_selection": False, "deployment_selection_rule": None,
        "claim_boundary": "Post-hoc diagnostic accuracy only. A truth-close sampled candidate does not establish a truth-blind selection rule, calibrated set coverage or improved point estimator."}
    save(destination, result)
    print(json.dumps([{k: r[k] for k in ("role", "seed", "ratio", "maximum_parameter_relative_error",
        "maximum_outer_coefficient_relative_error", "parameter_gate_pass", "outer_coefficient_gate_pass")} for r in records], indent=2))


if __name__ == "__main__":
    main()
