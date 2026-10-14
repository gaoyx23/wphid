import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r62_sparse_noise_boundary_has_unique_set_and_failure_tiers():
    path = ROOT / "artifacts/paper_r62_sparse_observation_noise_boundary/metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["confirmation_was_run"] is True
    assert payload["largest_tested_multiplier_with_four_of_four_truth_coverage"] == 0.025
    assert payload["largest_tested_multiplier_with_four_of_four_exact_recovery"] is None

    for level in ("0.01", "0.025"):
        aggregate = payload["results"][level]["confirmation_aggregate"]
        assert aggregate["all_five_truth_coverage_rate"] == 1.0
        assert aggregate["complete_exact_five_mechanism_rate"] == 0.0

    failed = payload["results"]["0.05"]["confirmation_aggregate"]
    assert failed["carbon_truth_coverage_rate"] == 1.0
    assert failed["crack_truth_coverage_rate"] == 0.0
    assert failed["transport_truth_coverage_rate"] == 0.75
    assert failed["all_five_truth_coverage_rate"] == 0.0
