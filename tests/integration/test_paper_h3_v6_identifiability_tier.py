import json
from pathlib import Path

import numpy as np

from experiments import paper_h3_v6_identifiability_tier as experiment


def test_v6_uses_new_unseen_seed_ranges_and_retains_v5():
    assert set(experiment.PROTOCOL["development_seeds"]).isdisjoint(
        experiment.PROTOCOL["confirmation_seeds"]
    )
    assert set(experiment.PROTOCOL["development_seeds"]).isdisjoint(
        range(42101, 42300)
    )
    assert "V6" in experiment.PROTOCOL["version"]
    assert experiment.PROTOCOL_HASH


def test_amplitude_stability_is_truth_blind_for_null_model():
    selected = {"family": None}
    result = experiment._amplitude_stability(selected, [], [], [])
    assert result["full_decomposition_stable"] is True
    assert result["mechanism_action_stable"] is True
    assert result["edges"] == []


def test_amplitude_stability_rejects_unstable_condition_coefficients(monkeypatch):
    class Fit:
        parameters = ((np.array([1.0, 1.0, 1.0]),),)
        coefficients = (np.array([1.0]),)

    selected = {
        "family": "fake",
        "fit": Fit(),
        "groups": (0,),
        "topology": (True,),
    }
    local = [object(), object(), object(), object()]
    bases = (np.empty((8, 0)),)
    targets = (np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]),)

    monkeypatch.setattr(experiment, "FIELDS", ("temperature",))
    monkeypatch.setattr(
        experiment.h3,
        "FAMILIES",
        {"fake": object()},
    )
    monkeypatch.setattr(
        experiment.h3,
        "_builder",
        lambda local, specification: lambda parameters: (np.ones(8),),
    )
    result = experiment._amplitude_stability(
        selected, local, bases, targets
    )
    assert result["full_decomposition_stable"] is False
    assert result["minimum_sign_consistency"] == 0.5


def test_run_split_resumes_a_valid_existing_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "ROLES", ("shared",))
    monkeypatch.setitem(experiment.PROTOCOL, "development_seeds", [42301])
    record = {"split": "development", "role": "shared", "seed": 42301}
    output = tmp_path / "development-shared-42301.json"
    output.write_text(json.dumps({
        "protocol_sha256": experiment.PROTOCOL_HASH,
        "record": record,
    }), encoding="utf-8")

    assert experiment.run_split("development", tmp_path, jobs=1) == [record]


def test_action_tier_abstains_only_when_both_instability_signals_are_large():
    base = {
        "full_decomposition_stable": False,
        "mechanism_action_stable": True,
    }
    assert experiment._decide_output_tier(
        "family", "T|O|D", True, base
    ) == "mechanism_action"
    base["mechanism_action_stable"] = False
    assert experiment._decide_output_tier(
        "family", "T|O|D", True, base
    ) == "abstain"


def test_frozen_v2_artifact_retains_independent_coverage_no_go():
    payload = json.loads(Path(experiment.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
    assert payload["aggregate"]["decision"] == "NO-GO"
    assert payload["aggregate"]["wrong_output_gate_pass"] is True
    independent = next(
        row for row in payload["confirmation_summary"]
        if row["role"] == "independent"
    )
    assert independent["endpoint_success_rate"] == 1.0
    assert independent["correct_output_rate"] == 0.6
    assert independent["wrong_output_rate"] == 0.0
    assert independent["abstention_rate"] == 0.4
