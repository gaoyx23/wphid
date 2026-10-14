"""Saved numerical contracts; not recovery/coverage qualification."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/_audits/h3_latent_correction"


def audit():
    return json.loads((OUT / "audit.json").read_text(encoding="utf-8"))


def test_fixed_scope_no_parameter_refits_or_truth():
    saved = audit()
    assert saved["complete"] and len(saved["records"]) == 10
    assert not saved["truth_used"]
    assert saved["parameter_optimizations"] == saved["new_noise_draws"] == saved["new_physical_trajectories"] == 0
    for role in ("partial", "independent"):
        rows = [r for r in saved["records"] if r["role"] == role]
        assert {r["ratio"] for r in rows} == {.5, .75, 1., 1.25, 1.5}
        assert all(len(r["packets"]) == 8 for r in rows)


def test_frozen_source_chain():
    for path, expected in audit()["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_minimum_linear_energy_identity_and_nonlinear_recheck_recorded():
    for row in audit()["records"]:
        assert row["all_function_domains_preserved"]
        np.testing.assert_allclose(row["correction_energy"], row["original_quadratic"], rtol=1e-7)
        for packet in row["packets"]:
            assert packet["function_domain_preserved"]
            assert np.isfinite(packet["nonlinear_remaining_whitened_norm_relative"])
            assert packet["nonlinear_remaining_quadratic_under_original_covariance"] >= 0
            assert packet["linear_balance_remaining_relative"] < 1e-7
