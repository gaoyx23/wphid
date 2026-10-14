import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r81_locates_failure_in_reporting_refit_not_family_selection():
    payload = _read(
        "artifacts/paper_r81_short_bell_eiv_failure_diagnostic/metrics.json"
    )
    assert payload["candidate_ranking_complete"] is True
    assert payload["missing_candidate_families"] == []
    assert payload["raw_identified_set"] == ["combined_opening_truth"]
    failures = payload["fit_diagnostics"]["failures"]
    assert len(failures) == 1
    assert failures[0]["phase"] == "all_conditions_refit"
    assert failures[0]["family"] == "crack_only"
    assert failures[0]["species"] == "pO2"


def test_r82_fresh_development_gate_passes_after_decoupling():
    payload = _read(
        "artifacts/paper_r82_selection_refit_decoupling_development/development.json"
    )
    baseline = payload["results"]["baseline"]
    eiv = payload["results"]["selection_refit_decoupled_eiv"]
    assert baseline["identified_set_size"] == 3
    assert eiv["identified_set"] == ["combined_opening_truth"]
    assert eiv["ranked_family_count"] == 6
    assert eiv["loocv_fit_failure_count"] == 0
    assert payload["development_gate"] is True


def test_r83_short_bell_locked_confirmation_is_complete_and_unique():
    payload = _read(
        "artifacts/paper_r83_short_bell_decoupled_eiv_confirmation/metrics.json"
    )
    aggregate = payload["aggregate"]
    assert aggregate["replicate_count"] == 4
    assert aggregate["structure_complete_eiv_count"] == 4
    assert aggregate["fallback_count"] == 0
    assert aggregate["guarded_truth_coverage_count"] == 4
    assert aggregate["guarded_unique_truth_structure_count"] == 4
    assert aggregate["truth_family_coefficients_reportable_count"] == 4
    assert aggregate["all_five_truth_coverage_count"] == 4
    assert aggregate["mean_baseline_set_size"] == 3.0
    assert aggregate["mean_guarded_set_size"] == 1.0
