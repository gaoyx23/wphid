import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r32_cross_solver_transport_confirmation/metrics.json"


def test_r32_cross_solver_confirmation_artifact():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["condition_count"] == 13
    assert all(
        row["finite_nonnegative"]
        for row in payload["solver_diagnostics"].values()
    )
    assert len(payload["noisy"]["locked_confirmation"]) == 4
    assert payload["noise_level_relative_to_nominal"] == 0.005

