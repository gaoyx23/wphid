"""Re-evaluate saved likelihood gradients, with zero optimization steps.

Finite-budget termination is not evidence of stationarity. This utility checks
the declared scaled/box-constrained first-order condition, not global quality.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from experiments.run_h3_joint_likelihood_development import OUT, build_model
from experiments.run_h3_moment_precision_development import OUT as PARENT


def projected_gradient(vector, gradient, kinetic_count, tolerance=1e-10):
    result = np.asarray(gradient).copy()
    for i in range(kinetic_count):
        if (vector[i] <= tolerance and result[i] > 0) or (vector[i] >= 1-tolerance and result[i] < 0):
            result[i] = 0
    return result


def main():
    path = OUT / "audit.json"
    if not path.exists():
        raise SystemExit("Full likelihood batch required; no optimization started.")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit["complete"]:
        raise RuntimeError("completed audit required")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for source, expected in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen source changed: {source}")
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), path, PARENT / "audit.json")}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "stationarity.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8"))["signature"] == signature:
        print("Completed stationarity check; no evaluations repeated.")
        return
    parent = json.loads((PARENT / "audit.json").read_text(encoding="utf-8"))
    references = {(r["role"], r["seed"]): r for r in parent["records"] if r["operator"] == "moment_corrected_degree4"}
    torch.set_num_threads(4)
    rows = []
    for saved in audit["records"]:
        model, _, _ = build_model(references[(saved["role"], saved["seed"])], "cpu")
        parameters = np.asarray(saved["parameters"][0])
        coefficients = np.asarray(saved["coefficients"])
        vector = np.concatenate((((parameters-model.lower.numpy())/model.width.numpy()).ravel(),
                                 (coefficients/model.coefficient_scale.numpy()).ravel()))
        value, gradient = model.value_and_gradient(vector)
        if not np.isclose(value, saved["final_nll"], rtol=1e-7, atol=1e-8):
            raise RuntimeError("CPU reconstructed likelihood differs from saved result")
        projected = projected_gradient(vector, gradient, model.kinetic_count)
        norm = float(np.max(np.abs(projected)))
        rows.append({"role": saved["role"], "seed": saved["seed"], "nll": value,
            "projected_scaled_gradient_infinity_norm": norm,
            "meets_declared_gtol": norm <= audit["protocol"]["gtol"],
            "projected_scaled_gradient": projected.tolist(), "optimizer_success": saved["optimizer_success"],
            "interpretation": "First-order check at saved fit in declared optimization coordinates; no global/parameter accuracy claim."})
        print(f"{saved['role']} {saved['seed']}: projected gradient {norm:.6g}", flush=True)
        del model
    result = {"complete": True, "signature": signature, "source_sha256": hashes,
        "records": rows, "optimizations": 0, "gradient_evaluations": len(rows), "device": "cpu",
        "meets_declared_gtol": sum(r["meets_declared_gtol"] for r in rows)}
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
