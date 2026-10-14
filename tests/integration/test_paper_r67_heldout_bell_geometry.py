import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r67_fixed_layout_covers_truth_on_heldout_bell_geometry():
    path = ROOT / "artifacts/paper_r67_heldout_bell_geometry_confirmation/metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    qualification = payload["qualification"]
    assert qualification["length_scale"] == 0.95
    assert qualification["monotone_bell"] is True
    assert qualification["all_conditions_finite"] is True
    assert qualification["all_peak_temperatures_below_2200_k"] is True
    assert payload["observation_design"]["meridional_sensor_indices"] == [0, 5, 15, 30]
    assert payload["development_gate_truth_coverage_scoring_only"] is True
    assert payload["confirmation_was_run"] is True
    aggregate = payload["confirmation_aggregate"]
    assert aggregate["replicate_count"] == 4
    assert aggregate["carbon_truth_coverage_rate"] == 1.0
    assert aggregate["crack_truth_coverage_rate"] == 1.0
    assert aggregate["transport_truth_coverage_rate"] == 1.0
    assert aggregate["all_five_truth_coverage_rate"] == 1.0
    assert aggregate["complete_exact_five_mechanism_rate"] == 0.0
    for row in payload["confirmation"]:
        assert row["carbon"]["exact_recovery_scoring_only"] is True
        assert row["crack"]["exact_recovery_scoring_only"] is True
