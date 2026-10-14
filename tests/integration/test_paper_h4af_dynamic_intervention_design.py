import json
from pathlib import Path

from experiments import paper_h4af_dynamic_intervention_design as design


def test_dynamic_design_retains_standard_panel_and_frozen_gate():
    assert design.PANELS["standard"] == ()
    assert design.PROTOCOL["minimum_relative_projected_distance"] == 0.10
    assert design.PROTOCOL["truth_use"].startswith("projected-geometry scoring only")


def test_retained_dynamic_design_matches_protocol():
    payload = json.loads(Path(design.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == design.PROTOCOL_HASH
    assert set(payload["panels"]) == set(design.PANELS)
