import json
from pathlib import Path


def _load(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def test_transport_uniqueness_requires_all_eight_thickness_locations():
    root = Path(__file__).resolve().parents[2]
    r23 = _load(
        root,
        "artifacts/paper_r23_thickness_resolution_boundary/metrics.json",
    )

    assert r23["minimum_unique_thickness_count_scoring_only"] == 8
    assert not r23["results"]["3"]["unique_truth_recovery_scoring_only"]
    assert r23["results"]["8"]["unique_truth_recovery_scoring_only"]


def test_temporal_boundary_reports_nonmonotone_selection_conservatively():
    root = Path(__file__).resolve().parents[2]
    r24 = _load(
        root,
        "artifacts/paper_r24_temporal_resolution_boundary/metrics.json",
    )

    assert r24["nonmonotonic_selection_detected"]
    assert r24[
        "largest_cadence_with_all_finer_tested_cadences_successful_scoring_only"
    ] == 6
    assert r24["results"]["6"][
        "complete_exact_five_mechanism_recovery_scoring_only"
    ]
    assert not r24["results"]["12"][
        "complete_exact_five_mechanism_recovery_scoring_only"
    ]


def test_raw_point_noise_exposes_crack_and_transport_bottlenecks():
    root = Path(__file__).resolve().parents[2]
    r25 = _load(
        root,
        "artifacts/paper_r25_fullfield_noise_stability/metrics.json",
    )

    assert r25["aggregate"]["0.0"]["complete_exact_five_mechanism_rate"] == 1.0
    assert r25["aggregate"]["0.001"]["carbon_exact_rate"] == 1.0
    assert r25["aggregate"]["0.001"]["crack_exact_rate"] == 0.0
    assert r25["aggregate"]["0.001"]["transport_unique_truth_rate"] == 1.0
    assert r25["aggregate"]["0.05"]["all_five_truth_coverage_rate"] == 0.0


def test_physics_preserving_reconstruction_improves_but_does_not_hide_limit():
    root = Path(__file__).resolve().parents[2]
    r27 = _load(
        root,
        "artifacts/paper_r27_physics_preserving_noise_validation/metrics.json",
    )

    assert r27["aggregate"]["0.001"][
        "complete_exact_five_mechanism_rate"
    ] == 1.0
    assert r27["aggregate"]["0.005"]["carbon_exact_rate"] == 1.0
    assert r27["aggregate"]["0.005"]["crack_exact_rate"] == 0.0
    assert r27["aggregate"]["0.005"]["transport_unique_truth_rate"] == 0.0
