import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r65_throat_focused_layout_restores_crack_truth_at_0p025():
    path = ROOT / "artifacts/paper_r65_throat_focused_sensor_layout/metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        assert row["crack"]["exact_recovery_scoring_only"] is True
