import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r47_cross_scale_excitation_screened_confirmation/metrics.json"


def test_r47_confirms_topology_but_withholds_alpha():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["screen_unchanged_from_R46"]["included_species"] == [
        "pO2",
        "pH2O",
    ]
    assert len(payload["locked_confirmation"]) == 4
    aggregate = payload["aggregate"]
    assert aggregate["confirmation_truth_coverage_count"] == 4
    assert aggregate["confirmation_unique_truth_count"] == 3
    assert aggregate["confirmation_alpha_quantitatively_identifiable_count"] == 0
    assert all(
        row["alpha_output"] == "not_quantitatively_identifiable"
        for row in payload["locked_confirmation"]
    )
