import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r35_guarded_transport_decision_confirmation/metrics.json"


def test_r35_guarded_transport_decision_is_correct_on_locked_panels():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    assert aggregate["locked_confirmation_correct_rate"] == 1.0
    assert aggregate["locked_in_library_true_acceptance_rate"] == 1.0
    assert aggregate["locked_out_of_library_rejection_rate"] == 1.0
    assert aggregate["locked_false_rejection_count"] == 0
    assert aggregate["locked_false_acceptance_count"] == 0

