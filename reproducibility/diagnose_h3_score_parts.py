"""Frozen-point score and noise-scale audit; zero optimizations/new noise."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from experiments.run_h3_nonlinear_amplitude_profile import OUT as PROFILE, protocol
from experiments.run_h3_curvature_continuation import OUT as START
from experiments.run_h3_joint_likelihood_development import build_model
from experiments.run_h3_moment_precision_development import OUT as PARENT, save, h3
from reproducibility.diagnose_h3_likelihood_curvature import CASES, saved_vector
from wphid.discovery.likelihood_score_parts import score_parts
from wphid.discovery.weak_bootstrap import centered_measurement_bootstrap

OUT = ROOT / "artifacts/_audits/h3_score_parts"
RULE = {"cases": CASES, "optimizations": 0, "new_noise_draws": 0,
        "scope": "saved points only; zero-added-noise surrogate is not zero declared noise variance",
        "clean_fields": "existing manufactured fields only for post-fit noise-scale audit; not parameter selection",
        "stop": "two cases, saved four profile points each; no altered score fitting"}


def main():
    torch.set_num_threads(4)
    protocol()
    paths = [Path(__file__), PROFILE / "audit.json", START / "audit.json", PARENT / "audit.json",
             ROOT / "src/wphid/discovery/likelihood_score_parts.py",
             ROOT / "src/wphid/discovery/weak_bootstrap.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps([hashes, RULE], sort_keys=True).encode()).hexdigest()
    final = OUT / "audit.json"
    if final.exists():
        if json.loads(final.read_text())["signature"] != signature:
            raise RuntimeError("preserve frozen score diagnostic")
        print("Completed score-parts audit; no evaluations repeated.")
        return
    save(OUT / "protocol.json", {"signature": signature, "source_sha256": hashes, "rule": RULE})
    refs = {(r["role"], r["seed"]): r for r in json.loads((PARENT / "audit.json").read_text())["records"]
            if r["operator"] == "moment_corrected_degree4"}
    starts = json.loads((START / "audit.json").read_text())["records"]
    profiles = json.loads((PROFILE / "audit.json").read_text())["records"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for role, seed in CASES:
        model, _, _ = build_model(refs[(role, seed)], device)
        fitted = next(r for r in starts if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
        vector = saved_vector(model, fitted)
        baseline = score_parts(model, vector)
        total_gradient = np.asarray(baseline["logdet_gradient"]) + baseline["quadratic_gradient"]
        value, gradient = model.value_and_gradient(vector)
        if not np.allclose(gradient, total_gradient, atol=1e-8, rtol=1e-8):
            raise RuntimeError("gradient decomposition mismatch")
        centered = centered_measurement_bootstrap(model, vector, 0, noise_multiplier=0.)
        zero = score_parts(centered, vector)
        points = []
        for r in [r for r in profiles if (r["role"], r["seed"]) == (role, seed)]:
            p = score_parts(model, saved_vector(model, {**r, "parameters": [r["parameters"]]}))
            dl = model.residual_count * (p["normalized_half_logdet"] - baseline["normalized_half_logdet"])
            dq = model.residual_count * (p["normalized_half_quadratic"] - baseline["normalized_half_quadratic"])
            if not np.isclose(dl+dq, r["unnormalized_nll_difference"], atol=1e-7):
                raise RuntimeError("saved profile score mismatch")
            points.append({"ratio": r["ratio"], "half_logdet_change": dl,
                           "half_quadratic_change": dq, "total_score_change": dl+dq})
        noise = []
        for name, packet in zip(h3.h1.CONDITIONS, model.packets):
            clean = np.asarray(h3._clean(name, role)[3]).reshape(3, -1)
            observed = packet["states"].cpu().numpy()
            actual_sigma = h3.PROTOCOL["noise_fraction"] * np.sqrt(np.mean(clean**2, axis=1))
            fitted_sigma = packet["noise"].cpu().numpy()
            z = (observed-clean)/actual_sigma[:, None]
            noise.append({"condition": name, "fit_sigma_to_generation_sigma": (fitted_sigma/actual_sigma).tolist(),
                          "standardized_noise_mean": z.mean(axis=1).tolist(),
                          "standardized_noise_std": z.std(axis=1).tolist()})
        row = {"role": role, "seed": seed, "baseline": baseline,
               "total_gradient_norm": float(np.linalg.norm(total_gradient)),
               "zero_added_noise_reference": zero,
               "profile_score_components": points, "post_fit_noise_scale_audit": noise}
        records.append(row)
        save(OUT / f"{role}-{seed}.json", {"signature": signature, **row})
        print(json.dumps({"role": role, "seed": seed, "profile_components": points,
                          "zero_quadratic": zero["normalized_half_quadratic"],
                          "zero_logdet_gradient_norm": float(np.linalg.norm(zero["logdet_gradient"]))}), flush=True)
        del model, centered
    save(final, {"signature": signature, "source_sha256": hashes, "complete": True,
                "optimizations": 0, "new_noise_draws": 0, "new_physical_trajectories": 0,
                "records": records, "claim_boundary": "Score decomposition is not evidence that logdet should be deleted or that its gradient is estimator bias. A centered zero-added-noise realization retains positive declared covariance. No new estimator/accuracy/coverage claim."})


if __name__ == "__main__":
    main()
