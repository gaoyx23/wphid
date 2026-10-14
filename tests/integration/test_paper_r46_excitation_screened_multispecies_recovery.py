import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r46_excitation_screened_multispecies_recovery/metrics.json"


def test_r46_screen_is_truth_blind_and_preserves_all_controls():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    screen = payload["screen_frozen_before_evaluation"]
    assert screen["truth_alpha_used_for_screen"] is False
    assert screen["included_species"] == ["pO2", "pH2O"]
    assert screen["excluded_species"] == ["pProducts"]
    assert payload["candidate_alpha_grid_unchanged"] == [
        0.0,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]
    aggregate = payload["aggregate"]
    assert aggregate["in_library_isotropic"][
        "confirmation_truth_coverage_count"
    ] == 4
    assert aggregate["out_of_library_anisotropic"][
        "confirmation_unique_truth_count"
    ] == 4
    assert aggregate["out_of_library_Darcy_Forchheimer"][
        "confirmation_truth_coverage_count"
    ] == 4
    assert aggregate["out_of_library_Darcy_Forchheimer"][
        "confirmation_unique_truth_count"
    ] == 3
    assert aggregate["out_of_library_Darcy_Forchheimer"][
        "confirmation_internal_isotropic_alpha_minimum_count"
    ] == 0
