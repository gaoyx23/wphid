"""Post-continuation local quadratic scales, never calibrated intervals.

No optimizations. Check the exact criterion at two curvature-predicted points
to expose where a local quadratic approximation itself stops being useful.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from experiments.run_h3_curvature_continuation import OUT as SOURCE, freeze
from experiments.run_h3_joint_likelihood_development import build_model
from experiments.run_h3_moment_precision_development import OUT as PARENT, save
from reproducibility.diagnose_h3_likelihood_curvature import CASES, saved_vector
from wphid.discovery.curvature_coordinates import finite_difference_hessian, curvature_summary

OUT = ROOT / "artifacts/_audits/h3_curvature_continuation"


def main():
    torch.set_num_threads(4)
    freeze()  # verifies continuation dependencies without any fits
    paths = [Path(__file__), SOURCE / "audit.json", ROOT / "src/wphid/discovery/curvature_coordinates.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "local-resolution.json"
    if destination.exists():
        if json.loads(destination.read_text())["signature"] != signature:
            raise RuntimeError("preserve previous local diagnostic")
        print("Completed local resolution diagnostic; no evaluations repeated.")
        return
    prior = json.loads((PARENT / "audit.json").read_text())
    refs = {(r["role"], r["seed"]): r for r in prior["records"] if r["operator"] == "moment_corrected_degree4"}
    audit = json.loads((SOURCE / "audit.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for role, seed in CASES:
        saved = next(r for r in audit["records"] if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
        model, _, _ = build_model(refs[(role, seed)], device)
        vector = saved_vector(model, saved)
        value, _ = model.value_and_gradient(vector)
        hessian, asymmetry = finite_difference_hessian(model.value_and_gradient, vector)
        summary = curvature_summary(hessian)
        row = {"role": role, "seed": seed, "nll": value, "summary": summary,
               "relative_raw_asymmetry": asymmetry, "optimizations": 0}
        if summary["positive_definite"]:
            inverse = np.linalg.inv(hessian)
            equation = 1 if role == "independent" else 2
            coordinate = model.kinetic_count+4*equation+3
            # Joint compensating direction for an unnormalized NLL rise of .5.
            direction = inverse[:, coordinate]/np.sqrt(model.residual_count*inverse[coordinate, coordinate])
            amplitude = saved["coefficients"][equation][3]
            relative = abs(direction[coordinate]*model.coefficient_scale.cpu().numpy()[equation, 3]/amplitude)
            actual = []
            for sign in (-1, 1):
                point = vector+sign*direction
                feasible = bool(np.all(point[:model.kinetic_count] >= 0) and np.all(point[:model.kinetic_count] <= 1))
                actual.append({"sign": sign, "kinetic_box_feasible": feasible,
                    "exact_unnormalized_nll_rise": (model.value_and_gradient(point)[0]-value)*model.residual_count if feasible else None})
            row.update({"amplitude_field": ("oxygen" if equation == 1 else "damage"),
                "joint_quadratic_half_unit_nll_amplitude_relative_change": relative,
                "compensating_direction_in_optimization_coordinates": direction.tolist(),
                "predicted_unnormalized_nll_rise": .5, "exact_checks": actual,
                "interpretation": "Local joint quadratic geometry only, not posterior uncertainty, calibrated confidence bounds or proof of non-identifiability; exact checks may invalidate linear extrapolation."})
        records.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("summary", "compensating_direction_in_optimization_coordinates")}), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    save(destination, {"signature": signature, "source_sha256": hashes, "complete": True,
                       "records": records, "optimizations": 0, "truth_used": False})


if __name__ == "__main__":
    main()
