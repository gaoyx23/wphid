import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r60_meridional_boundary_is_four_locations():
    payload = _load(
        "artifacts/paper_r60_meridional_resolution_boundary/metrics.json"
    )
    assert payload[
        "minimum_count_with_all_denser_tested_counts_covering_truth_scoring_only"
    ] == 4
    assert payload[
        "minimum_count_with_all_denser_tested_counts_exact_scoring_only"
    ] == 4
    assert payload["results"]["4"][
        "complete_exact_five_mechanism_recovery_scoring_only"
    ] is True
    assert payload["results"]["5"][
        "complete_exact_five_mechanism_recovery_scoring_only"
    ] is True
    assert payload["results"]["3"][
        "all_five_truth_mechanisms_covered_scoring_only"
    ] is False
    assert payload["results"]["3"]["crack_truth_covered_scoring_only"] is False
    assert payload["results"]["3"]["carbon_truth_covered_scoring_only"] is True
    assert payload["results"]["3"]["transport_truth_covered_scoring_only"] is True


def test_r61_sparse_asynchronous_confirmation_is_locked_and_complete():
    payload = _load(
        "artifacts/paper_r61_sparse_asynchronous_seven_field_confirmation/metrics.json"
    )
    assert payload["development_gate_truth_coverage_scoring_only"] is True
    assert payload["confirmation_was_run"] is True
    design = payload["observation_design"]
    assert design["meridional_sensor_count"] == 5
    assert design["missing_fraction"] == 0.05
    assert design["jitter_fraction"] == 0.20
    aggregate = payload["confirmation_aggregate"]
    assert aggregate["replicate_count"] == 4
    assert aggregate["all_five_truth_coverage_rate"] == 1.0
    assert aggregate["complete_exact_five_mechanism_rate"] == 1.0


def test_r63_minimum_spatial_design_is_confirmed():
    payload = _load(
        "artifacts/paper_r63_minimum_spatial_observation_confirmation/metrics.json"
    )
    assert payload["observation_design"]["meridional_sensor_count"] == 4
    assert payload["development_gate_truth_coverage_scoring_only"] is True
    assert payload["confirmation_was_run"] is True
    aggregate = payload["confirmation_aggregate"]
    assert aggregate["replicate_count"] == 4
    assert aggregate["all_five_truth_coverage_rate"] == 1.0
    assert aggregate["complete_exact_five_mechanism_rate"] == 1.0
