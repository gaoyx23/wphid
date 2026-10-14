import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r84_reference_and_long_bells_preserve_truth_with_partial_uniqueness():
    payload = _read(
        "artifacts/paper_r84_corrected_eiv_reference_long_confirmation/metrics.json"
    )
    reference = payload["results"]["reference_bell"]["aggregate"]
    long_bell = payload["results"]["long_bell_plus5pct"]["aggregate"]
    assert reference["guarded_truth_coverage_count"] == 4
    assert reference["guarded_unique_truth_structure_count"] == 1
    assert reference["fallback_count"] == 2
    assert long_bell["guarded_truth_coverage_count"] == 4
    assert long_bell["guarded_unique_truth_structure_count"] == 2
    assert long_bell["fallback_count"] == 2


def test_r85_is_the_no_new_fit_corrected_three_geometry_summary():
    payload = _read(
        "artifacts/paper_r85_corrected_three_geometry_summary/metrics.json"
    )
    combined = payload["combined"]
    assert payload["new_model_fits"] == 0
    assert payload["seed_sets_disjoint"] is True
    assert combined["replicate_count"] == 12
    assert combined["guarded_truth_coverage_count"] == 12
    assert combined["all_five_truth_coverage_count"] == 12
    assert combined["guarded_unique_truth_structure_count"] == 7
    assert combined["truth_family_coefficients_reportable_count"] == 8
    assert combined["fallback_count"] == 4
    assert combined["mean_guarded_set_size"] < combined["mean_baseline_set_size"]
