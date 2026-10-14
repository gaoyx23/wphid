import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r56_missing_directional_control_exposes_guard_bias():
    payload = _load(
        "artifacts/paper_r56_guarded_multiregime_transport_routing/development.json"
    )
    assert payload["development_gate_passed"] is False
    assert payload["aggregate"]["forchheimer_strong"]["truth_coverage_count"] == 1
    directional = payload["aggregate"]["forchheimer_strong_directional"]
    assert directional["truth_coverage_count"] == 0
    assert directional["wrong_only_count"] == 1


def test_r57_ratio_profile_does_not_resolve_strong_directional_truth():
    payload = _load(
        "artifacts/paper_r57_profiled_directional_forchheimer_topology/development.json"
    )
    assert payload["development_gate_passed"] is False
    directional = payload["aggregate"]["forchheimer_strong_directional"]
    assert directional["truth_coverage_count"] == 0
    assert directional["wrong_only_count"] == 1


def test_r58_locked_identified_set_covers_both_strong_topologies():
    payload = _load(
        "artifacts/paper_r58_strong_nonlinear_topology_identified_set/metrics.json"
    )
    assert payload["confirmation_gate_passed"] is True
    for panel in ("strong_isotropic", "strong_directional"):
        block = payload["aggregate"][panel]
        assert block["truth_coverage_count"] == 4
        assert block["unique_truth_count"] == 0
        assert block["wrong_only_count"] == 0
        assert block["topology_disagreement_count"] == 4

