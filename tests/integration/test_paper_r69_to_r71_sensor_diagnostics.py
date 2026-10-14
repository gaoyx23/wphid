import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r69_added_downstream_sensor_fails_development_gate():
    payload = _read(
        "artifacts/paper_r69_long_bell_nested_sensor_augmentation/development.json"
    )
    assert payload["observation_design"]["meridional_sensor_indices"] == [0, 5, 15, 22, 30]
    assert payload["development_gate_truth_coverage_scoring_only"] is False
    assert payload["confirmation_was_run"] is False
    assert payload["development"]["carbon"]["truth_covered_scoring_only"] is True
    assert payload["development"]["crack"]["truth_covered_scoring_only"] is True
    assert payload["development"]["transport"]["truth_covered_scoring_only"] is False


def test_r70_is_labeled_seed_aligned_not_exactly_paired():
    payload = _read(
        "artifacts/paper_r70_paired_long_bell_sensor_diagnostic/metrics.json"
    )
    assert payload["observation_pairing"] == "seed-aligned_only_not_value-paired"
    assert payload["aggregate"] == {
        "four_sensor_transport_truth_coverage_count": 3,
        "five_sensor_transport_truth_coverage_count": 3,
        "rescued_count": 1,
        "degraded_count": 1,
    }


def test_r71_exact_pairing_finds_no_information_gain_from_index_22():
    payload = _read(
        "artifacts/paper_r71_exact_nested_sensor_pairing/metrics.json"
    )
    assert payload["sampling_parent_indices"] == [0, 5, 15, 22, 30]
    aggregate = payload["aggregate"]
    assert aggregate["paired_count"] == 5
    assert aggregate["four_sensor_transport_truth_coverage_count"] == 3
    assert aggregate["five_sensor_transport_truth_coverage_count"] == 3
    assert aggregate["rescued_count"] == 0
    assert aggregate["degraded_count"] == 0
    assert aggregate["both_cover_count"] == 3
    assert aggregate["neither_covers_count"] == 2
    for row in payload["records"]:
        assert row["four_sensor_transport_set"] == row["five_sensor_transport_set"]


def test_r72_attributes_the_long_bell_failure_to_noise_not_missingness():
    payload = _read(
        "artifacts/paper_r72_long_bell_imperfection_attribution/metrics.json"
    )
    records = payload["records"]
    assert records["asynchronous_complete_clean"]["transport_identified_set"] == [
        "combined_opening_truth"
    ]
    assert records["asynchronous_missing5_clean"]["transport_identified_set"] == [
        "combined_opening_truth"
    ]
    assert records["asynchronous_complete_noise0p025"][
        "transport_truth_covered"
    ] is False
    assert records["asynchronous_missing5_noise0p025"][
        "transport_truth_covered"
    ] is False
    assert records["asynchronous_complete_noise0p025"][
        "transport_identified_set"
    ] == records["asynchronous_missing5_noise0p025"]["transport_identified_set"]
