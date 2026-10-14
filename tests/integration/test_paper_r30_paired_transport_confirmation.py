import json
from pathlib import Path


def _load(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def test_paired_transport_confirmation_sets_half_percent_complete_boundary():
    root = Path(__file__).resolve().parents[2]
    r27 = _load(
        root,
        "artifacts/paper_r27_physics_preserving_noise_validation/metrics.json",
    )
    r28 = _load(
        root,
        "artifacts/paper_r28_integral_crack_noise_validation/metrics.json",
    )
    r30 = _load(
        root,
        "artifacts/paper_r30_paired_transport_confirmation/metrics.json",
    )

    assert r27["aggregate"]["0.005"]["carbon_exact_rate"] == 1.0
    assert r28["aggregate"]["0.005"]["confirmation_exact_recovery_rate"] == 1.0
    assert r30["aggregate"]["0.005"]["confirmation_unique_truth_rate"] == 1.0

    assert r28["aggregate"]["0.01"]["confirmation_exact_recovery_rate"] == 1.0
    assert r30["aggregate"]["0.01"]["confirmation_minimum_error_truth_rate"] == 1.0
    assert r30["aggregate"]["0.01"]["confirmation_truth_coverage_rate"] == 1.0
    assert r30["aggregate"]["0.01"]["confirmation_unique_truth_rate"] == 0.0
