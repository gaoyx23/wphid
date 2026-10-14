"""Frozen saved evidence checks; no refits or gradient evaluations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/_audits/h3_joint_likelihood_development"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_complete_batch_keeps_all_capped_fits_and_known_noise_qualification():
    audit = read("audit.json")
    assert audit["complete"] and len(audit["records"]) == 8
    assert audit["conditional_refits"] == 8
    assert audit["new_noise_realizations"] == audit["new_structure_searches"] == 0
    assert not audit["protocol"]["official_WENDy_implementation"]
    assert "calibrated" in audit["protocol"]["additional_input"]
    assert all(r["iterations"] == 200 and not r["optimizer_success"] for r in audit["records"])
    assert [r["quantitative_pass"] for r in audit["summary"]] == [3, 3]
    assert read("status.json")["state"] == "completed" and not (OUT / "worker.lock").exists()


def test_frozen_likelihood_and_analysis_sources_match():
    for filename in ("protocol.json", "comparison.json", "stationarity.json"):
        for path, expected in read(filename)["source_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_paired_improvements_are_not_claimed_as_new_passes_or_convergence():
    paired = read("comparison.json")["paired"][-1]
    assert paired["improved_maximum_parameter_relative_error"] == 7
    assert paired["improved_maximum_outer_coefficient_relative_error"] == 5
    assert paired["quantitative_pass_gained"] == paired["quantitative_pass_lost"] == 0
    stationarity = read("stationarity.json")
    assert stationarity["complete"] and stationarity["optimizations"] == 0
    assert stationarity["gradient_evaluations"] == 8 and stationarity["meets_declared_gtol"] == 0
