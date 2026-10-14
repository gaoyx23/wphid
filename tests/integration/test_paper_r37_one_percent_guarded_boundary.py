import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r37_one_percent_guarded_transport_boundary/metrics.json"


def test_r37_uses_disjoint_adequacy_calibration_and_locked_evaluation():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["noise_level_relative_to_nominal"] == 0.01
    assert payload["adequacy_calibration"]["disjoint_from_evaluation_seeds"]
    assert len(payload["adequacy_calibration"]["rows"]) == 5
    assert len(payload["panels"]) == 3
    assert all(
        len(block["locked_confirmation"]) == 4
        for block in payload["panels"].values()
    )

