import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r40_expanded_transport_open_set_guard/metrics.json"


def test_r40_uses_disjoint_calibration_and_reports_all_three_panels():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["noise_level_relative_to_nominal"] == 0.01
    assert payload["adequacy_calibration"]["disjoint_from_evaluation_seeds"]
    assert set(payload["results"]) == {
        "in_library_isotropic",
        "out_of_library_anisotropic",
        "out_of_library_Darcy_Forchheimer",
    }
    assert all(
        len(block["locked_confirmation"]) == 4
        for block in payload["results"].values()
    )
    assert all(
        "adequacy_pass" in row and "panel_decision_correct" in row
        for block in payload["results"].values()
        for row in block["locked_confirmation"]
    )
    aggregate = payload["aggregate"]
    assert aggregate["in_library_isotropic"]["adequacy_acceptance_count"] == 4
    assert aggregate["out_of_library_anisotropic"]["adequacy_acceptance_count"] == 4
    assert (
        aggregate["out_of_library_Darcy_Forchheimer"][
            "adequacy_acceptance_count"
        ]
        == 0
    )
    assert aggregate["all_panels"]["panel_decision_correct_count"] == 12
    assert aggregate["all_panels"]["unsafe_forchheimer_family_output_count"] == 0
