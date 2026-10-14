import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r66_0p05_development_failure_is_retained():
    path = ROOT / "artifacts/paper_r66_throat_focused_noise_limit/development.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["development_gate_truth_coverage_scoring_only"] is False
    assert payload["confirmation_was_run"] is False
    assert payload["development"]["all_five_truth_mechanisms_covered_scoring_only"] is False
    assert payload["development"]["crack"]["truth_covered_scoring_only"] is False
    assert payload["development"]["crack"]["selected_support"] == [
        "stress_threshold_truth"
    ]
