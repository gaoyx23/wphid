"""Saved profile provenance and no-oracle-selection checks, no refits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/_audits/h3_nonlinear_amplitude_profile"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_exact_declared_grid_budgets_and_constraints_retained():
    audit = read("audit.json")
    assert audit["complete"] and len(audit["records"]) == audit["conditional_nuisance_fits"] == 8
    assert audit["existing_observation_packets"] == 2 and audit["new_noise_realizations"] == 0
    for role in ("partial", "independent"):
        rows = [r for r in audit["records"] if r["role"] == role]
        assert {r["ratio"] for r in rows} == {.5, .75, 1.25, 1.5}
        assert len({r["input_packet_sha256"] for r in rows}) == 1
        assert all(r["objective_evaluations"] <= 150 and r["iterations"] <= 60 for r in rows)
        assert all(r["optimizer_success"] and not r["any_kinetic_box_active"] for r in rows)
        equation = 1 if role == "independent" else 2
        assert all(abs(r["coefficients"][equation][3]-r["fixed_amplitude"]) < 1e-8 for r in rows)
    assert not (OUT / "worker.lock").exists()


def test_frozen_sources_and_diagnostic_qualification():
    for filename in ("protocol.json", "analysis.json", "truth-scoring.json"):
        for path, expected in read(filename)["source_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert read("analysis.json")["optimizations"] == 0 and not read("analysis.json")["truth_used"]
    truth = read("truth-scoring.json")
    assert not truth["truth_used_for_point_selection"] and truth["deployment_selection_rule"] is None


def test_truth_close_competitors_exist_but_not_turned_into_an_estimator():
    rows = read("truth-scoring.json")["records"]
    oxygen = next(r for r in rows if r["role"] == "independent" and r["ratio"] == .75)
    damage = next(r for r in rows if r["role"] == "partial" and r["ratio"] == 1.25)
    for row in (oxygen, damage):
        assert row["parameter_gate_pass"] and row["outer_coefficient_gate_pass"]
        assert row["unnormalized_nll_difference"] < 1
    assert "truth-blind selection rule" in read("truth-scoring.json")["claim_boundary"]
