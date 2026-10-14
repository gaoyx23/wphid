import json
from pathlib import Path


def test_unified_confirmation_is_seed_aligned_and_preserves_set_boundary():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((
        root
        / "artifacts/paper_r31_unified_noisy_five_mechanism_confirmation/metrics.json"
    ).read_text(encoding="utf-8"))

    assert payload["aggregate"]["0.005"]["confirmation_all_five_exact_count"] == 4
    assert payload["aggregate"]["0.005"]["confirmation_all_five_exact_rate"] == 1.0
    assert payload["aggregate"]["0.01"]["confirmation_all_five_exact_count"] == 0
    assert payload["aggregate"]["0.01"][
        "confirmation_four_local_exact_transport_set_count"
    ] == 4
