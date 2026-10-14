"""Saved profile score costs and observed-action/anchor changes, no fits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from experiments.run_h3_nonlinear_amplitude_profile import OUT, protocol
from experiments.run_h3_curvature_continuation import OUT as START
from experiments.run_h3_joint_likelihood_development import build_model
from experiments.run_h3_moment_precision_development import OUT as PARENT, save
from reproducibility.diagnose_h3_likelihood_curvature import CASES
from wphid.discovery.centered_reaction import observed_pivot, encode_strength, centered_action


def main():
    torch.set_num_threads(4)
    protocol()
    paths = [Path(__file__), OUT / "audit.json", START / "audit.json",
             ROOT / "src/wphid/discovery/centered_reaction.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "analysis.json"
    if destination.exists():
        if json.loads(destination.read_text())["signature"] != signature:
            raise RuntimeError("preserve completed analysis")
        print("Completed saved nonlinear profile analysis; no fits/evaluations repeated.")
        return
    audit = json.loads((OUT / "audit.json").read_text())
    if not audit["complete"] or len(audit["records"]) != 8:
        raise RuntimeError("all eight profile checkpoints required")
    parent = json.loads((PARENT / "audit.json").read_text())
    refs = {(r["role"], r["seed"]): r for r in parent["records"] if r["operator"] == "moment_corrected_degree4"}
    starts = json.loads((START / "audit.json").read_text())
    results = []
    for role, seed in CASES:
        saved = next(r for r in starts["records"] if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
        model, _, _ = build_model(refs[(role, seed)], "cpu")
        equation = 1 if role == "independent" else 2
        group = model.groups[equation]
        states = np.concatenate([p["states"].numpy() for p in model.packets], axis=1)
        pivot = observed_pivot(states)
        parameters = np.asarray(saved["parameters"][0])[group]
        amplitude = saved["coefficients"][equation][3]
        sign, strength = encode_strength(amplitude, parameters, pivot)
        base_action = centered_action(states, sign, strength, parameters, pivot)
        points = []
        for row in [r for r in audit["records"] if (r["role"], r["seed"]) == (role, seed)]:
            if row["input_packet_sha256"] != saved["input_packet_sha256"]:
                raise RuntimeError("profile observation identity mismatch")
            p = np.asarray(row["parameters"])[group]
            a = row["coefficients"][equation][3]
            if not np.isclose(a, row["fixed_amplitude"], atol=1e-8, rtol=1e-10):
                raise RuntimeError("saved amplitude equality violated")
            s, k = encode_strength(a, p, pivot)
            action = centered_action(states, s, k, p, pivot)
            weak_original, weak_profile = [], []
            offset = 0
            for packet in model.packets:
                length = packet["states"].shape[1]
                mass = packet["mass"].numpy()
                weak_original.extend(mass@base_action[offset:offset+length])
                weak_profile.extend(mass@action[offset:offset+length])
                offset += length
            weak_original, weak_profile = np.asarray(weak_original), np.asarray(weak_profile)
            points.append({"amplitude_ratio": row["ratio"], "unnormalized_nll_difference": row["unnormalized_nll_difference"],
                "optimizer_success": row["optimizer_success"], "kinetic_box_active": row["any_kinetic_box_active"],
                "equality_tangent_coordinate_gradient_norm": row["equality_tangent_coordinate_gradient_norm"],
                "anchor_strength_relative_change": abs(float(s/sign*np.exp(k-strength))-1),
                "observed_nodal_action_relative_change": float(np.linalg.norm(action-base_action)/np.linalg.norm(base_action)),
                "observed_weak_action_relative_change": float(np.linalg.norm(weak_profile-weak_original)/np.linalg.norm(weak_original)),
                "kinetic_parameters": p.tolist()})
        result = {"role": role, "seed": seed, "field": "oxygen" if equation == 1 else "damage",
            "fixed_pivot_log_features": pivot.tolist(), "reference_kinetic_parameters": parameters.tolist(), "points": points,
            "interpretation": "Changes relative to the saved fit on the same noisy observed states, not errors relative to ground truth; the pivot is only an exact coordinate reference, not new physical information."}
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        del model
    save(destination, {"complete": True, "signature": signature, "source_sha256": hashes,
        "records": results, "optimizations": 0, "truth_used": False,
        "claim_boundary": "Sparse local nuisance profile grid, one start each, approximate first-order likelihood; not calibrated confidence intervals, posterior probabilities or global-minimum guarantees."})


if __name__ == "__main__":
    main()
