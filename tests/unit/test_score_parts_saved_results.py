"""Read-only saved-point audit contracts, not scientific success gates."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/_audits/h3_score_parts"


def audit():
    return json.loads((OUT / "audit.json").read_text(encoding="utf-8"))


def test_no_fit_or_noise_expansion_and_frozen_inputs():
    saved = audit()
    assert saved["complete"] and len(saved["records"]) == 2
    assert saved["optimizations"] == saved["new_noise_draws"] == saved["new_physical_trajectories"] == 0
    for path, expected in saved["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_score_parts_add_and_zero_added_noise_keeps_covariance():
    for row in audit()["records"]:
        assert {p["ratio"] for p in row["profile_score_components"]} == {.5, .75, 1.25, 1.5}
        for p in row["profile_score_components"]:
            assert np.isclose(p["half_logdet_change"] + p["half_quadratic_change"], p["total_score_change"])
        zero = row["zero_added_noise_reference"]
        assert zero["normalized_half_quadratic"] < 1e-20
        assert np.linalg.norm(zero["quadratic_gradient"]) < 1e-8
        np.testing.assert_allclose(zero["logdet_gradient"], row["baseline"]["logdet_gradient"], rtol=1e-10, atol=1e-10)


def test_clean_noise_audit_only_descriptive_not_an_estimator():
    saved = audit()
    assert "not evidence that logdet should be deleted" in saved["claim_boundary"]
    for row in saved["records"]:
        assert len(row["post_fit_noise_scale_audit"]) == 8
        for noise in row["post_fit_noise_scale_audit"]:
            assert len(noise["fit_sigma_to_generation_sigma"]) == 3
            assert np.all(np.isfinite(noise["standardized_noise_mean"]))
            assert np.all(np.asarray(noise["standardized_noise_std"]) > 0)
