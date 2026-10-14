"""Fixed extra test functions, same data: conditional information, zero fits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.run_h3_nonlinear_amplitude_profile import protocol
from experiments.run_h3_curvature_continuation import OUT as START
from experiments.run_h3_moment_precision_development import OUT as PARENT, save, h3, corrected_problem, groups_from_saved_partition
from reproducibility.diagnose_h3_likelihood_curvature import CASES
from wphid.weak.moment_corrected import MomentCorrectedAxisymmetricProjector
from wphid.weak.polynomial_tests import PolynomialTestProjector, MODES
from wphid.discovery.weak_mean_information import whitened_mean_jacobian, mean_information_summary

OUT = ROOT / "artifacts/_audits/h3_test_enrichment"


def main():
    protocol()
    paths = [Path(__file__), START / "audit.json", PARENT / "audit.json",
             ROOT / "src/wphid/weak/polynomial_tests.py", ROOT / "src/wphid/discovery/weak_mean_information.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / "audit.json"
    if final.exists():
        if json.loads(final.read_text())["signature"] != signature:
            raise RuntimeError("preserve diagnostic")
        print("Completed test enrichment diagnostic; no evaluations repeated.")
        return
    save(OUT / "protocol.json", {"signature": signature, "source_sha256": hashes, "modes": MODES,
        "cases": CASES, "new_observations": 0, "optimizations": 0,
        "covariance": "full window/test/equation shared-noise covariance, not independent added rows",
        "screen": "conditional amplitude scale proxy improves at least 20% and clean balance relative error below 1e-4 before any fitting",
        "truth": "only clean manufactured-balance check after information analysis; not test selection, weighting or fitted-point sensitivity"})
    parent = json.loads((PARENT / "audit.json").read_text())
    refs = {(r["role"], r["seed"]): r for r in parent["records"] if r["operator"] == "moment_corrected_degree4"}
    starts = json.loads((START / "audit.json").read_text())
    results = []
    with threadpool_limits(limits=4):
        for role, seed in CASES:
            reference = refs[(role, seed)]
            saved = next(r for r in starts["records"] if (r["role"], r["seed"], r["variant"]) == (role, seed, "curvature_coordinates"))
            local, _, _ = h3._assemble(seed, role)
            identity = corrected_problem(local, seed, role)[3]
            if identity != saved["input_packet_sha256"]:
                raise RuntimeError("observation packet mismatch")
            parameters, coefficients = np.asarray(saved["parameters"][0]), np.asarray(saved["coefficients"])
            groups = groups_from_saved_partition(reference["source_selected_partition"])
            equation = 1 if role == "independent" else 2
            coordinate = parameters.size+4*equation+3
            variants = []
            for name, projector_class in (("original_tests", MomentCorrectedAxisymmetricProjector), ("four_fixed_tests", PolynomialTestProjector)):
                blocks, clean_residuals, clean_actions = [], [[] for _ in range(3)], [[] for _ in range(3)]
                for condition, (states, _, _) in zip(h3.h1.CONDITIONS, local):
                    t, z, r, clean, forcing = h3._clean(condition, role)
                    projector = projector_class(t=t, z=z, r=r, window_shape=tuple(h3.h1.PROTOCOL["window_shape"]),
                         stride_shape=tuple(h3.h1.PROTOCOL["stride_shape"]), degree=4)
                    libraries = [projector.library(u, f) for u, f in zip(states, forcing)]
                    noise = [h3.PROTOCOL["noise_fraction"]*float(np.sqrt(np.mean(u*u))) for u in states]
                    blocks.append(whitened_mean_jacobian(states, projector.operators, [l[0] for l in libraries],
                                  parameters, coefficients, groups, noise))
                    # Truth is used only for independent clean numerical balance.
                    temperature, oxygen, damage = clean
                    for e in range(3):
                        theta, m, q = h3.TRUTH_CASES[role]["parameters"][e]
                        action = h3.TRUTH_AMPLITUDES[e]*projector.mass(np.exp(-theta/temperature)*oxygen**m*(1-damage)**q)
                        base, target = projector.library(clean[e], forcing[e])
                        actual = np.asarray(h3.h1.CLOSED["truth_coefficients"][h3.FIELDS[e]][:-1])
                        clean_residuals[e].extend(target-base@actual-action)
                        clean_actions[e].extend(action)
                summary = mean_information_summary(np.vstack(blocks), coordinate)
                summary.update(variant=name, weak_tests_per_condition=projector.operators["mass"].shape[0],
                    clean_balance_relative_errors=[float(np.linalg.norm(a)/np.linalg.norm(b)) for a, b in zip(clean_residuals, clean_actions)])
                variants.append(summary)
            old, new = variants
            ratio = new["amplitude_relative_scale_proxy"]/old["amplitude_relative_scale_proxy"] if old["full_numerical_rank"] and new["full_numerical_rank"] else None
            row = {"role": role, "seed": seed, "input_packet_sha256": identity, "variants": variants,
                   "enriched_to_original_amplitude_scale_proxy_ratio": ratio,
                   "worth_minimal_fit_check": bool(ratio is not None and ratio <= .8 and max(new["clean_balance_relative_errors"]) < 1e-4)}
            results.append(row)
            print(json.dumps({"role": role, "seed": seed, "ratio": ratio,
                "old_proxy": old["amplitude_relative_scale_proxy"], "new_proxy": new["amplitude_relative_scale_proxy"],
                "clean_errors": new["clean_balance_relative_errors"], "worth_fit": row["worth_minimal_fit_check"]}), flush=True)
    save(final, {"signature": signature, "complete": True, "records": results, "optimizations": 0,
        "new_observations": 0, "claim_boundary": "Mean-only local information at fitted parameters and clean manufactured-balance diagnostics; not estimation accuracy, independent confirmation or confidence calibration."})


if __name__ == "__main__":
    main()
