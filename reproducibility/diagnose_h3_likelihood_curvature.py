"""Two saved difficult cases: local Hessian only, zero optimizations."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from experiments.run_h3_joint_likelihood_development import OUT as SOURCE, build_model
from experiments.run_h3_moment_precision_development import OUT as PARENT, save
from wphid.discovery.curvature_coordinates import finite_difference_hessian, curvature_summary

OUT = ROOT / "artifacts/_audits/h3_likelihood_curvature"
CASES = (("independent", 42303), ("partial", 42302))


def saved_vector(model, saved):
    return np.concatenate((((np.asarray(saved["parameters"][0])-model.lower.cpu().numpy())/model.width.cpu().numpy()).ravel(),
                           (np.asarray(saved["coefficients"])/model.coefficient_scale.cpu().numpy()).ravel()))


def main():
    torch.set_num_threads(4)
    frozen = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    for source, expected in frozen["source_sha256"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen source changed: {source}")
    paths = [Path(__file__), ROOT / "src/wphid/discovery/curvature_coordinates.py", SOURCE / "audit.json", PARENT / "audit.json"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / "audit.json"
    if final.exists():
        if json.loads(final.read_text())["signature"] != signature:
            raise RuntimeError("preserve existing diagnostic; source changed")
        print("Completed curvature diagnostic; no evaluations repeated.")
        return
    save(OUT / "protocol.json", {"signature": signature, "source_sha256": hashes,
        "cases": CASES, "step": 1e-5, "check_step": 5e-6,
        "optimizations": 0, "truth_used": False, "scope": "local finite-difference likelihood curvature, not confidence intervals or a global identifiability theorem"})
    parent = json.loads((PARENT / "audit.json").read_text())
    refs = {(r["role"], r["seed"]): r for r in parent["records"] if r["operator"] == "moment_corrected_degree4"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for role, seed in CASES:
        path = OUT / f"{role}-{seed}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["signature"] != signature:
                raise RuntimeError("diagnostic checkpoint mismatch")
        else:
            model, _, _ = build_model(refs[(role, seed)], device)
            saved = json.loads((SOURCE / "cases" / f"{role}-{seed}.json").read_text())
            vector = saved_vector(model, saved)
            value, gradient = model.value_and_gradient(vector)
            if not np.isclose(value, saved["final_nll"], atol=1e-8):
                raise RuntimeError("saved objective mismatch")
            matrix, asymmetry = finite_difference_hessian(model.value_and_gradient, vector)
            check, check_asymmetry = finite_difference_hessian(model.value_and_gradient, vector, step=5e-6)
            relative = float(np.linalg.norm(matrix-check)/np.linalg.norm(check))
            row = {"signature": signature, "role": role, "seed": seed, "nll": value,
                   "hessian": matrix.tolist(), "gradient": gradient.tolist(), "vector": vector.tolist(),
                   "kinetic_count": model.kinetic_count, "relative_step_check_difference": relative,
                   "relative_raw_asymmetry": asymmetry, "relative_check_asymmetry": check_asymmetry,
                   "summary": curvature_summary(matrix), "optimizations": 0}
            save(path, row)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        records.append(row)
        print(json.dumps({"role": role, "seed": seed, "step_difference": row["relative_step_check_difference"], **row["summary"]}), flush=True)
    save(final, {"signature": signature, "complete": True, "records": records, "optimizations": 0})


if __name__ == "__main__":
    main()
