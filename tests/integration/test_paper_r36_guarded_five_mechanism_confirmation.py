import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r36_guarded_five_mechanism_confirmation/metrics.json"


def test_r36_full_system_claim_is_safe_on_locked_panels():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    assert aggregate["locked_confirmation_correct_count"] == 12
    assert aggregate["locked_confirmation_panel_count"] == 12
    assert aggregate["in_library_all_five_exact_count"] == 4
    assert aggregate["out_of_library_local_four_exact_safe_transport_count"] == 8
    assert aggregate["unsafe_five_mechanism_claim_count_on_out_of_library"] == 0

