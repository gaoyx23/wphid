import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r51_forchheimer_strength_generalization/metrics.json"


def test_r51_exposes_weak_and_strong_nonlinearity_boundaries():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["frozen_before_new_strengths"]["strength_multipliers"] == [
        0.5,
        1.0,
        2.0,
    ]
    aggregate = payload["aggregate"]
    assert aggregate["0.5"]["truth_law_topology_coverage_count"] == 4
    assert aggregate["0.5"]["unique_truth_law_topology_count"] == 2
    assert aggregate["0.5"]["numeric_alpha_outputs"] == [0.25, 0.25]
    assert aggregate["1.0"]["truth_law_topology_coverage_count"] == 4
    assert aggregate["1.0"]["unique_truth_law_topology_count"] == 3
    assert aggregate["1.0"]["numeric_alpha_outputs"] == [0.5, 0.5, 0.5]
    assert aggregate["2.0"]["truth_law_topology_coverage_count"] == 0
    assert aggregate["2.0"]["numeric_alpha_output_count"] == 0
