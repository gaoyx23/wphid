import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r45_natural_condition_information_audit/metrics.json"


def test_r45_attributes_locked_passive_information_without_resimulation():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["reused_without_resimulation"]["condition_count"] == 13
    assert len(payload["condition_ranking"]) == 13
    assert set(payload["condition_evidence_by_seed"]) == {
        row["condition_id"] for row in payload["condition_ranking"]
    }
    assert all(
        len(rows) == 4
        for rows in payload["condition_evidence_by_seed"].values()
    )
    assert all(
        0 <= row["positive_law_evidence_seed_count"] <= 4
        and 0 <= row["positive_topology_evidence_seed_count"] <= 4
        and row["pooled_Forchheimer_number_p90"] >= 0.0
        and 0.0 <= row["pooled_flux_attenuation_p90"] <= 1.0
        for row in payload["condition_ranking"]
    )
