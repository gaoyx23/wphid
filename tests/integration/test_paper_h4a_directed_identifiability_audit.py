import json
from pathlib import Path

from experiments import paper_h4a_directed_identifiability_audit as experiment


def test_directed_identifiability_artifact_matches_protocol_and_condition_set():
    payload = json.loads(Path(experiment.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
    assert tuple(payload["protocol"]["condition_names"]) == tuple(
        experiment.h1.CONDITIONS
    )
    assert len(payload["results"]) == 1 + len(experiment.h1.CONDITIONS)


def test_directed_alias_is_closer_than_remaining_damage_family():
    payload = json.loads(Path(experiment.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    alternatives = payload["results"]["all_conditions"]["alternatives"]
    alias = alternatives["oxygen_power_damage"]
    other = alternatives["oxygen_remaining_power"]
    assert alias["relative_projected_distance"] < other["relative_projected_distance"]
    assert payload["diagnosis"]["literal_family_uniquely_resolved"] is False


def test_directed_family_is_exactly_aliased_at_unit_exponent():
    payload = json.loads(Path(experiment.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    rows = {
        row["truth_exponent"]: row
        for row in payload["truth_exponent_identifiability_sweep"]
    }
    assert rows[1.0]["relative_projected_distance"] < 1e-8
    assert rows[0.5]["relative_projected_distance"] > rows[0.9][
        "relative_projected_distance"
    ]
    assert rows[1.5]["relative_projected_distance"] > rows[1.1][
        "relative_projected_distance"
    ]
