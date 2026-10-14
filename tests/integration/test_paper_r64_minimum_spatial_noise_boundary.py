import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r64_minimum_spatial_noise_boundary_is_crack_limited():
    path = ROOT / "artifacts/paper_r64_minimum_spatial_noise_boundary/metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["confirmation_was_run"] is True
    assert payload["largest_tested_multiplier_with_four_of_four_truth_coverage"] == 0.01
    assert payload["largest_tested_multiplier_with_four_of_four_exact_recovery"] is None

    low = payload["results"]["0.01"]["confirmation_aggregate"]
    assert low["all_five_truth_coverage_rate"] == 1.0
    assert low["complete_exact_five_mechanism_rate"] == 0.0

    for level in ("0.025", "0.05"):
        aggregate = payload["results"][level]["confirmation_aggregate"]
        assert aggregate["carbon_truth_coverage_rate"] == 1.0
        assert aggregate["crack_truth_coverage_rate"] == 0.0
        assert aggregate["transport_truth_coverage_rate"] == 1.0
        assert aggregate["all_five_truth_coverage_rate"] == 0.0
