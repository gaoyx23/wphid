import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r59_asynchronous_missing_noisy_seven_field_confirmation():
    path = (
        ROOT
        / "artifacts/paper_r59_asynchronous_seven_field_observation_validation"
        / "metrics.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["development_gate_truth_coverage_scoring_only"] is True
    assert payload["confirmation_was_run"] is True

    asynchronous = payload["scenarios"]["asynchronous_clean_complete"][
        "aggregate"
    ]
    assert asynchronous["all_five_truth_coverage_rate"] == 1.0
    assert asynchronous["complete_exact_five_mechanism_rate"] == 1.0

    missing = payload["scenarios"]["asynchronous_clean_missing5"]["aggregate"]
    assert missing["all_five_truth_coverage_rate"] == 1.0
    assert missing["complete_exact_five_mechanism_rate"] == 0.75

    combined = payload["scenarios"][
        "asynchronous_missing5_noise0p005"
    ]["aggregate"]
    assert combined["replicate_count"] == 4
    assert combined["carbon_truth_coverage_rate"] == 1.0
    assert combined["crack_truth_coverage_rate"] == 1.0
    assert combined["transport_truth_coverage_rate"] == 1.0
    assert combined["all_five_truth_coverage_rate"] == 1.0
    assert combined["complete_exact_five_mechanism_rate"] == 1.0
