import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r52_single_vehicle_condition_budget/metrics.json"


def test_r52_quantifies_single_vehicle_information_budget_without_new_simulation():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["terminal_snapshot_only"][
        "numerically_evaluable_by_current_method"
    ] is False
    aggregate = payload["aggregate"]
    assert aggregate["recent_mission_plus_cooldown"][
        "truth_structure_coverage_count"
    ] == 4
    assert aggregate["recent_mission_plus_cooldown"][
        "numeric_alpha_outputs"
    ] == [0.25, 0.25]
    assert aggregate["recent_two_cycles_plus_cooldown"][
        "unique_truth_structure_count"
    ] == 0
    assert aggregate["four_cycle_history_plus_cooldown"][
        "unique_truth_structure_count"
    ] == 4
    assert aggregate["four_cycle_history_plus_cooldown"][
        "numeric_alpha_outputs"
    ] == [0.5, 1.0, 0.5, 0.5]
    assert aggregate["history_plus_basic_offline_envelope"][
        "numeric_alpha_outputs"
    ] == [0.5, 1.0, 1.0, 0.5]
    assert aggregate["full_offline_natural_envelope"][
        "unique_truth_structure_count"
    ] == 3
    assert aggregate["full_offline_natural_envelope"][
        "numeric_alpha_outputs"
    ] == [0.5, 0.5, 0.5]
    assert all(
        row["wrong_structure_exclusion_count"] == 0
        for row in aggregate.values()
    )
