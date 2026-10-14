import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r79_guarded_eiv_is_safe_but_geometry_dependent():
    payload = _read(
        "artifacts/paper_r79_cross_geometry_guarded_eiv_confirmation/metrics.json"
    )
    reference = payload["results"]["reference_bell"]["aggregate"]
    short = payload["results"]["short_bell_minus5pct"]["aggregate"]
    assert reference["guarded_truth_coverage_count"] == 4
    assert reference["guarded_unique_truth_count"] == 2
    assert reference["fallback_count"] == 2
    assert short["guarded_truth_coverage_count"] == 4
    assert short["guarded_unique_truth_count"] == 0
    assert short["fallback_count"] == 4


def test_r80_three_geometry_summary_uses_disjoint_seeds_and_no_new_fits():
    payload = _read(
        "artifacts/paper_r80_three_geometry_guarded_eiv_summary/metrics.json"
    )
    combined = payload["combined"]
    assert payload["new_model_fits"] == 0
    assert payload["seed_sets_disjoint"] is True
    assert combined["replicate_count"] == 12
    assert combined["guarded_truth_coverage_count"] == 12
    assert combined["all_five_truth_coverage_count"] == 12
    assert combined["guarded_unique_truth_count"] == 4
    assert combined["fallback_count"] == 8
    assert combined["mean_guarded_set_size"] < combined["mean_baseline_set_size"]
