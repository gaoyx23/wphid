import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r33_anisotropic_transport_open_set_audit/metrics.json"


def test_r33_is_frozen_realistic_open_set_control():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    control = payload["physical_negative_control"]
    assert control["meridional_to_through_diffusivity_ratio"] == 3.0
    assert payload["condition_count"] == 13
    assert len(payload["noisy"]["locked_confirmation"]) == 4
    assert all(
        row["finite_nonnegative"]
        for row in payload["trajectory_diagnostics"].values()
    )

