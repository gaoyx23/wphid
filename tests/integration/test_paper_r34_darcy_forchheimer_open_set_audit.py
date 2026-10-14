import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r34_darcy_forchheimer_open_set_audit/metrics.json"


def test_r34_uses_frozen_realistic_forchheimer_control():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    control = payload["physical_negative_control"]
    assert control["Forchheimer_number_at_reference_flux"] == 1.0
    assert control["flux_reference_percentile"] == 90.0
    assert payload["condition_count"] == 13
    assert len(payload["noisy"]["locked_confirmation"]) == 4
    assert all(
        row["finite_nonnegative"]
        for row in payload["trajectory_diagnostics"].values()
    )

