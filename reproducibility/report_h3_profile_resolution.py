"""Truth-blind sampled-profile resolution targets; zero scientific fits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.run_h3_nonlinear_amplitude_profile import OUT, protocol
from experiments.run_h3_curvature_continuation import OUT as START
from experiments.run_h3_moment_precision_development import save, h3
from reproducibility.diagnose_h3_likelihood_curvature import CASES
from wphid.discovery.profile_resolution import resolution_witness


def main():
    protocol()
    paths = [Path(__file__), OUT / "audit.json", START / "audit.json", ROOT / "src/wphid/discovery/profile_resolution.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / "resolution-witness.json"
    if destination.exists():
        if json.loads(destination.read_text())["signature"] != signature:
            raise RuntimeError("preserve resolution report")
        print("Completed profile resolution report; no computations repeated.")
        return
    profile = json.loads((OUT / "audit.json").read_text())
    starts = json.loads((START / "audit.json").read_text())
    if not profile["complete"]:
        raise RuntimeError("frozen complete profiles required")
    records = []
    tolerance = h3.PROTOCOL["outer_coefficient_relative_error_threshold"]
    for role, seed in CASES:
        reference = next(r for r in starts["records"] if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
        equation = 1 if role == "independent" else 2
        points = [{"ratio": 1., "amplitude": reference["coefficients"][equation][3], "score_cost": 0.}]
        points.extend({"ratio": r["ratio"], "amplitude": r["fixed_amplitude"], "score_cost": r["unnormalized_nll_difference"]}
            for r in profile["records"] if (r["role"], r["seed"]) == (role, seed))
        report = resolution_witness(points, tolerance)
        report.update(role=role, seed=seed, field="oxygen" if equation == 1 else "damage")
        records.append(report)
        print(json.dumps({"role": role, "score_budget": report["minimum_sampled_score_budget_for_incompatible_point_targets"]}), flush=True)
    save(destination, {"signature": signature, "source_sha256": hashes, "complete": True,
                       "records": records, "optimizations": 0, "truth_used": False, "statistical_budget_calibrated": False})


if __name__ == "__main__":
    main()
