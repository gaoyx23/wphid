"""Frozen pilot checks; no resampling or optimization."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/_audits/h3_score_bootstrap_pilot"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_declared_pilot_complete_without_expansion():
    audit = read("audit.json")
    assert audit["complete"] and audit["measurement_bootstrap_draws"] == 8
    assert audit["conditional_fits"] == 16 and audit["new_physical_trajectories"] == 0
    assert audit["valid_local_pairs"] == 8 and not audit["budget_calibrated"]
    rows = audit["records"]
    assert len(rows) == len({r["draw_seed"] for r in rows}) == 8
    assert all(not r["original_truth_used"] for r in rows)
    for row in rows:
        free, fixed = row["fits"]["unrestricted"], row["fits"]["fixed_reference_amplitude"]
        assert abs(row["profile_score_statistic_raw"] - 648 * (fixed["nll"] - free["nll"])) < 1e-12
        for fit in (free, fixed):
            assert fit["optimizer_success"] and fit["kinetic_box_feasible"]
            assert not fit["kinetic_box_active"]
            assert fit["objective_evaluations"] <= 150 and fit["iterations"] <= 60
            assert fit["fixed_amplitude_coordinate_error"] < 1e-8
    assert not (OUT / "worker.lock").exists()


def test_frozen_sources_still_match():
    for name in ("protocol.json", "summary.json"):
        for path, expected in read(name)["source_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_reference_variation_not_converted_to_coverage_or_accuracy():
    summary = read("summary.json")
    assert summary["complete"] and summary["new_optimizations"] == 0
    assert not summary["budget_calibrated"]
    for row in summary["records"]:
        assert row["count"] == 4 and row["empirical_95_percent_budget"] is None
        assert not row["uses_original_simulation_truth"]
        ratios = row["free_amplitude_to_plugin_reference_ratios"]
        assert sum(abs(r - 1) <= .25 for r in ratios) == row["free_amplitude_within_25_percent_of_plugin_reference"]
    assert read("audit.json")["protocol"]["theory_reference_only"]
