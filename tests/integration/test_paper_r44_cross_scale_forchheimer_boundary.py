import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r44_cross_scale_forchheimer_boundary/metrics.json"


def test_r44_separates_cross_scale_family_coverage_from_alpha_identifiability():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["alpha_grid_unchanged_from_R43"] == [
        0.0,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]
    assert payload["scales"]["intermediate"] != payload["scales"]["global"]
    assert len(payload["locked_confirmation"]) == 4
    assert all(
        "truth_law_topology_covered_scoring_only" in row
        and "alpha_quantitatively_identifiable" in row
        and "intermediate_isotropic_alpha_curve" in row
        and "global_isotropic_alpha_curve" in row
        for row in payload["locked_confirmation"]
    )
    aggregate = payload["aggregate"]
    assert aggregate["confirmation_truth_coverage_count"] == 3
    assert aggregate["confirmation_unique_truth_count"] == 0
    assert aggregate["confirmation_alpha_quantitatively_identifiable_count"] == 0
    assert aggregate["confirmation_intermediate_upper_boundary_count"] == 4
    assert aggregate["confirmation_global_upper_boundary_count"] == 0
