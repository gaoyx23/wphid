import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r49_dual_view_forchheimer_parameter_recovery/metrics.json"


def test_r49_gates_parameter_output_with_cross_scale_structure():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    workflow = payload["frozen_workflow"]
    assert workflow["selected_parameter_method"] == "raw_clipped"
    assert workflow["score_species"] == ["pO2", "pH2O"]
    aggregate = payload["aggregate"]
    assert aggregate["confirmation_structurally_unique_count"] == 3
    assert aggregate["confirmation_numeric_alpha_output_count"] == 3
    assert aggregate["confirmation_alpha_outputs"] == [0.5, 0.5, 0.5]
    assert aggregate[
        "confirmation_all_structurally_eligible_have_numeric_alpha"
    ] is True
    assert aggregate["confirmation_consistent_numeric_alpha"] is True
    assert all(
        row["alpha_output"] == "not_quantitatively_identifiable"
        and row["conditional_95_percent_alpha_set"]
        == "withheld_by_structural_gate"
        for row in payload["locked_confirmation"]
        if not row["structural_view_unique_Forchheimer_isotropic"]
    )
