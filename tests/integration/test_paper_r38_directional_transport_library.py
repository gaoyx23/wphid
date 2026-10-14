import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r38_directional_transport_library_recovery/metrics.json"


def test_r38_directional_library_reports_truth_coverage_and_uniqueness_separately():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["noise_level_relative_to_nominal"] == 0.01
    assert set(payload["results"]) == {
        "in_library_isotropic",
        "out_of_library_anisotropic",
    }
    assert all(
        len(block["locked_confirmation"]) == 4
        for block in payload["results"].values()
    )
    assert all(
        "truth_covered_scoring_only" in row
        and "unique_truth_recovery_scoring_only" in row
        for block in payload["results"].values()
        for row in block["locked_confirmation"]
    )

