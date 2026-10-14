import json
from pathlib import Path


def _load(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def test_integral_crack_recovery_extends_locked_noise_range_to_one_percent():
    root = Path(__file__).resolve().parents[2]
    r28 = _load(
        root,
        "artifacts/paper_r28_integral_crack_noise_validation/metrics.json",
    )

    for level in ("0.001", "0.005", "0.01"):
        assert r28["aggregate"][level]["confirmation_exact_recovery_rate"] == 1.0
    assert r28["aggregate"]["0.025"]["confirmation_exact_recovery_rate"] == 0.0


def test_repeated_use_history_preserves_truth_but_not_noisy_transport_uniqueness():
    root = Path(__file__).resolve().parents[2]
    r29 = _load(
        root,
        "artifacts/paper_r29_repeated_use_transport_identifiability/metrics.json",
    )

    assert all(r29["physical_checks"].values())
    assert r29["clean_result"]["unique_truth_recovery_scoring_only"]
    for level in ("0.005", "0.01"):
        assert r29["aggregate"][level]["confirmation_truth_coverage_rate"] == 1.0
        assert r29["aggregate"][level]["confirmation_unique_truth_rate"] == 0.0
        rows = r29["replicates"][level]["locked_confirmation"]
        assert all(row["selected_family"] == "combined_opening_truth" for row in rows)
