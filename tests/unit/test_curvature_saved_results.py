"""Completed continuation evidence checks, no scientific computation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURVATURE = ROOT / "artifacts/_audits/h3_likelihood_curvature"
CONTINUATION = ROOT / "artifacts/_audits/h3_curvature_continuation"


def read(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def test_local_curvature_checked_twice_but_not_an_extra_fit():
    audit = read(CURVATURE, "audit.json")
    assert audit["complete"] and len(audit["records"]) == 2 and audit["optimizations"] == 0
    assert all(r["relative_step_check_difference"] < 1e-5 and r["summary"]["positive_definite"] for r in audit["records"])
    assert all(r["summary"]["spectral_condition_if_positive"] > 1e8 for r in audit["records"])


def test_paired_continuations_keep_same_start_budget_data_and_failures():
    audit = read(CONTINUATION, "audit.json")
    assert audit["complete"] and len(audit["records"]) == 4
    assert audit["existing_observation_packets"] == 2 and audit["conditional_continuations"] == 4
    assert audit["new_noise_realizations"] == audit["new_structure_searches"] == 0
    for role in ("partial", "independent"):
        rows = [r for r in audit["records"] if r["role"] == role]
        assert rows[0]["initial_nll"] == rows[1]["initial_nll"]
        assert rows[0]["input_packet_sha256"] == rows[1]["input_packet_sha256"]
        assert all(r["iterations"] <= 100 and r["objective_evaluations"] <= 250 for r in rows)
        assert all(not r["outer_coefficient_gate_pass"] for r in rows)
    result = read(CONTINUATION, "comparison.json")
    assert all(v == 0 for v in result["new_quantitative_passes_vs_saved_start"].values())
    assert not (CONTINUATION / "worker.lock").exists()


def test_frozen_sources_unchanged():
    for directory, filename in ((CURVATURE, "protocol.json"), (CONTINUATION, "protocol.json"), (CONTINUATION, "comparison.json")):
        for path, expected in read(directory, filename)["source_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_local_quadratic_extrapolation_is_not_reported_as_a_confidence_interval():
    diagnostic = read(CONTINUATION, "local-resolution.json")
    assert diagnostic["complete"] and diagnostic["optimizations"] == 0 and not diagnostic["truth_used"]
    for row in diagnostic["records"]:
        assert "not posterior uncertainty" in row["interpretation"]
        assert all(check["exact_unnormalized_nll_rise"] > 10*row["predicted_unnormalized_nll_rise"] for check in row["exact_checks"])
    for path, expected in diagnostic["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
