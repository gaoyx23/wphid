"""Ten frozen-point latent correction prechecks; no parameter optimization."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import torch
from experiments.run_h3_nonlinear_amplitude_profile import OUT as PROFILE, protocol
from experiments.run_h3_curvature_continuation import OUT as START
from experiments.run_h3_joint_likelihood_development import build_model
from experiments.run_h3_moment_precision_development import OUT as PARENT, save
from reproducibility.diagnose_h3_likelihood_curvature import CASES, saved_vector
from wphid.discovery.latent_balance_correction import correction_summary

OUT = ROOT / "artifacts/_audits/h3_latent_correction"
RULE = {"cases": CASES, "points_per_case": 5, "parameter_optimizations": 0,
        "correction": "one minimum-norm linearized correction per condition with original calibrated noise",
        "no": "no iteration, truth, prior, physical trajectory, smoothing or new noise",
        "screening": "report nonlinear remaining whitened norm; no inference threshold or automatic estimator promotion"}


def main():
    torch.set_num_threads(4)
    protocol()
    paths = [Path(__file__), PROFILE / "audit.json", START / "audit.json", PARENT / "audit.json",
             ROOT / "src/wphid/discovery/latent_balance_correction.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps([hashes, RULE], sort_keys=True).encode()).hexdigest()
    final = OUT / "audit.json"
    if final.exists():
        if json.loads(final.read_text())["signature"] != signature:
            raise RuntimeError("preserve completed latent correction")
        print("Completed latent correction precheck; no linear solves repeated.")
        return
    save(OUT / "protocol.json", {"signature": signature, "source_sha256": hashes, "rule": RULE})
    refs = {(r["role"], r["seed"]): r for r in json.loads((PARENT / "audit.json").read_text())["records"]
            if r["operator"] == "moment_corrected_degree4"}
    starts = json.loads((START / "audit.json").read_text())["records"]
    profiles = json.loads((PROFILE / "audit.json").read_text())["records"]
    records = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for role, seed in CASES:
        model, _, _ = build_model(refs[(role, seed)], device)
        fitted = next(r for r in starts if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
        points = [(1., fitted)] + [(r["ratio"], {**r, "parameters": [r["parameters"]]})
            for r in profiles if (r["role"], r["seed"]) == (role, seed)]
        for ratio, point in points:
            path = OUT / f"{role}-{seed}-{ratio:g}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["signature"] != signature:
                    raise RuntimeError("checkpoint mismatch")
            else:
                result = correction_summary(model, saved_vector(model, point))
                row = {"signature": signature, "role": role, "seed": seed, "ratio": ratio, **result}
                save(path, row)
            records.append(row)
            print(json.dumps({"role": role, "ratio": ratio,
                  "energy": row["correction_energy"], "quadratic": row["original_quadratic"],
                  "max_nonlinear_remaining": max(p.get("nonlinear_remaining_whitened_norm_relative", float("inf")) for p in row["packets"])}), flush=True)
        del model
    save(final, {"signature": signature, "source_sha256": hashes, "complete": True,
                "records": records, "parameter_optimizations": 0, "new_noise_draws": 0,
                "new_physical_trajectories": 0, "truth_used": False,
                "claim_boundary": "Fixed parameter state-correction precheck only; no fitted latent fields/parameter improvement or new estimator."})


if __name__ == "__main__":
    main()
