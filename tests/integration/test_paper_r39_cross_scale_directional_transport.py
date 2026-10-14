import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    ROOT
    / "artifacts/paper_r39_cross_scale_directional_transport_confirmation/metrics.json"
)


def test_r39_reports_cross_scale_truth_coverage_and_abstention_separately():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["noise_level_relative_to_nominal"] == 0.01
    assert set(payload["results"]) == {
        "in_library_isotropic",
        "out_of_library_anisotropic",
    }
    assert payload["scales"]["intermediate"] != payload["scales"]["global"]
    assert all(
        len(block["locked_confirmation"]) == 4
        for block in payload["results"].values()
    )
    assert all(
        row["decision"] in {
            "scale_inconsistent_abstention",
            "unique_cross_scale_family",
            "cross_scale_candidate_set",
        }
        for block in payload["results"].values()
        for row in block["locked_confirmation"]
    )
    isotropic = payload["aggregate"]["in_library_isotropic"]
    anisotropic = payload["aggregate"]["out_of_library_anisotropic"]
    assert isotropic["clean_decision"] == "scale_inconsistent_abstention"
    assert isotropic["confirmation_truth_coverage_count"] == 4
    assert isotropic["confirmation_unique_truth_count"] == 4
    assert anisotropic["clean_decision"] == "unique_cross_scale_family"
    assert anisotropic["confirmation_truth_coverage_count"] == 4
    assert anisotropic["confirmation_unique_truth_count"] == 2
