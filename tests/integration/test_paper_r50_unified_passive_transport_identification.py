import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r50_unified_passive_transport_identification/metrics.json"


def test_r50_unifies_safe_locked_transport_outputs():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["workflow"]["passive_observations_only"] is True
    assert payload["workflow"]["new_natural_conditions_added"] is False
    aggregate = payload["aggregate"]
    assert aggregate["all_panel_seed_truth_coverage_count"] == 12
    assert aggregate["all_panel_seed_count"] == 12
    assert aggregate["Forchheimer_truth_coverage_count"] == 4
    assert aggregate["Forchheimer_unique_truth_count"] == 3
    assert aggregate["Forchheimer_numeric_alpha_output_count"] == 3
    assert aggregate[
        "Forchheimer_reported_alpha_sets_cover_truth_interval_count"
    ] == 3
    comparison = payload["comparison_with_R43"]
    assert comparison["R43_Forchheimer_truth_coverage_count"] == 3
    assert comparison["R43_Forchheimer_unique_truth_count"] == 0
    assert comparison["R50_Forchheimer_truth_coverage_count"] == 4
    assert comparison["R50_Forchheimer_unique_truth_count"] == 3
