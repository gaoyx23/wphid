"""Evaluate one omitted noise-mean term on frozen fits, zero optimizations."""
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
from reproducibility.diagnose_h3_likelihood_curvature import CASES, OUT as CURVATURE, saved_vector
from wphid.discovery.weak_mean_bias import MeanBiasDiagnostic

RULE = {"no_refit_if_all": {"estimated_local_amplitude_shift_below": .01,
                            "absolute_profile_score_change_below": .01},
        "scope": "practical screening for this one mean term, not a scientific recovery threshold or general bias bound"}


def main():
    torch.set_num_threads(4)
    protocol()
    paths = [Path(__file__), OUT / "audit.json", START / "audit.json", CURVATURE / "audit.json",
             ROOT / "src/wphid/discovery/weak_mean_bias.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps([hashes, RULE], sort_keys=True).encode()).hexdigest()
    destination = OUT / "mean-bias-diagnostic.json"
    if destination.exists():
        if json.loads(destination.read_text())["signature"] != signature:
            raise RuntimeError("preserve completed mean-bias diagnostic")
        print("Completed mean-bias diagnostic; no evaluations repeated.")
        return
    save(OUT / "mean-bias-protocol.json", {"signature": signature, "source_sha256": hashes,
        "screening_rule": RULE, "optimizations": 0,
        "approximation": "second-order IID Gaussian nonlinear-rate mean only; first-order covariance unchanged; plug-in observed states"})
    parent = json.loads((PARENT / "audit.json").read_text())
    refs = {(r["role"], r["seed"]): r for r in parent["records"] if r["operator"] == "moment_corrected_degree4"}
    prior = json.loads((START / "audit.json").read_text())
    profile = json.loads((OUT / "audit.json").read_text())
    curvature = json.loads((CURVATURE / "audit.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for role, seed in CASES:
        baseline = next(r for r in prior["records"] if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
        model, _, _ = build_model(refs[(role, seed)], device)
        vector = saved_vector(model, baseline)
        wrapper = MeanBiasDiagnostic(model)
        old, original_gradient = model.value_and_gradient(vector)
        corrected, corrected_gradient = wrapper.value_and_gradient(vector)
        hessian = np.asarray(next(r["hessian"] for r in curvature["records"] if (r["role"], r["seed"]) == (role, seed)))
        shift = -np.linalg.solve(hessian, corrected_gradient-original_gradient)
        equation = 1 if role == "independent" else 2
        coordinate = model.kinetic_count+4*equation+3
        amplitude_shift = abs(shift[coordinate]*model.coefficient_scale.cpu().numpy()[equation, 3]/baseline["coefficients"][equation][3])
        squared_correction, squared_action = np.zeros(3), np.zeros(3)
        with torch.no_grad():
            kinetics, coefficients = model.unpack(model.tensor(vector))
            for packet in model.packets:
                correction = wrapper.packet_correction(kinetics, coefficients, packet)
                temperature, oxygen, damage = packet["states"]
                theta, m, q = kinetics[model.group_index].unbind(dim=1)
                rate = torch.exp(-theta[:, None]/temperature)*oxygen**m[:, None]*(1-damage)**q[:, None]
                action = coefficients[:, 3, None]*(rate@packet["mass"].T)
                squared_correction += torch.sum(correction**2, dim=1).cpu().numpy()
                squared_action += torch.sum(action**2, dim=1).cpu().numpy()
        points = []
        for row in [r for r in profile["records"] if (r["role"], r["seed"]) == (role, seed)]:
            point = saved_vector(model, {**row, "parameters": [row["parameters"]]})
            value, _ = wrapper.value_and_gradient(point)
            new_cost = (value-corrected)*model.residual_count
            points.append({"ratio": row["ratio"], "original_cost": row["unnormalized_nll_difference"],
                "corrected_cost_at_same_saved_parameters": new_cost,
                "cost_change": new_cost-row["unnormalized_nll_difference"],
                "nuisance_reoptimized": False})
        result = {"role": role, "seed": seed,
            "relative_weak_mean_correction_by_equation": np.sqrt(squared_correction/squared_action).tolist(),
            "baseline_unnormalized_nll_change": (corrected-old)*model.residual_count,
            "estimated_local_amplitude_shift_relative": float(amplitude_shift),
            "local_shift_qualification": "linear response using previous saved-point Hessian, not a new fit or guaranteed bias correction",
            "profile_points": points,
            "do_not_refit_this_direction": bool(amplitude_shift < .01 and all(abs(p["cost_change"]) < .01 for p in points))}
        records.append(result)
        print(json.dumps(result, indent=2), flush=True)
        del model, wrapper
        if device == "cuda":
            torch.cuda.empty_cache()
    save(destination, {"signature": signature, "source_sha256": hashes, "complete": True,
        "records": records, "optimizations": 0, "truth_used": False, "screening_rule": RULE,
        "claim_boundary": "Only the specified plug-in second-order mean term is assessed. Does not exclude all EIV bias, covariance misspecification or finite-sample estimation bias; not confidence calibration."})


if __name__ == "__main__":
    main()
