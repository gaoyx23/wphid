import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r54_failure_is_retained():
    payload = _load(
        "artifacts/paper_r54_operator_consistent_forchheimer_topology/metrics.json"
    )
    aggregate = payload["aggregate"]["operator_consistent_diffusivity"]
    assert payload["development_gate_passed"] is True
    assert payload["confirmation_gate_passed"] is False
    assert aggregate["forchheimer_strong"]["truth_coverage_count"] == 0
    assert aggregate["forchheimer_strong"]["wrong_only_count"] == 4


def test_r55_nested_topology_locked_confirmation():
    payload = _load(
        "artifacts/paper_r55_nested_topology_forchheimer_recovery/metrics.json"
    )
    aggregate = payload["aggregate"]
    assert payload["confirmation_gate_passed"] is True
    assert aggregate["linear_isotropic"]["unique_truth_count"] == 4
    assert aggregate["linear_directional"]["unique_truth_count"] == 4
    assert aggregate["forchheimer_weak"]["unique_truth_count"] == 4
    assert aggregate["forchheimer_moderate"]["truth_coverage_count"] == 4
    assert aggregate["forchheimer_strong"]["truth_coverage_count"] == 4
    assert aggregate["forchheimer_strong"]["unique_truth_count"] == 0
    assert sum(block["wrong_only_count"] for block in aggregate.values()) == 0

