import json
from pathlib import Path

from experiments import paper_h4ae_excitation_design as experiment


def test_excitation_panels_expand_oxygen_and_damage_independently():
    grid = experiment.EXCITATION_PANELS["wide_independent_grid"]
    assert len(grid) == 9
    assert {row[1] for row in grid.values()} == {0.35, 1.0, 2.2}
    assert {row[2] for row in grid.values()} == {0.0, 0.35, 0.7}


def test_excitation_artifact_matches_frozen_development_protocol():
    payload = json.loads(Path(experiment.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
    assert set(payload["panels"]) == set(experiment.EXCITATION_PANELS)
    assert payload["aggregate"]["best_panel"] in experiment.EXCITATION_PANELS
