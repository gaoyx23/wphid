import json
from pathlib import Path

from experiments import paper_h4_complete_mechanism_noise_atlas as experiment


def test_h4_protocol_has_complete_presence_partition_and_noise_coverage():
    rows = experiment.build_scenario_matrix()
    h4a = [row for row in rows if row["phase"] == "H4-A"]
    assert len(h4a) == 8 * len(experiment.NOISE_MULTIPLIERS)
    assert {tuple(row["presence"].values()) for row in h4a} == set(
        experiment.PRESENCE_PATTERNS
    )
    h4b = [row for row in rows if row["phase"] == "H4-B"]
    assert {row["reaction_partition"] for row in h4b} == set(
        experiment.REACTION_PARTITIONS
    )
    h4c = [row for row in rows if row["phase"] == "H4-C"]
    assert {row["required_target"] for row in h4c} == {
        "complete_mechanism_system", "abstain"
    }


def test_h4_seed_splits_are_new_and_disjoint():
    audit = experiment.validate_protocol()
    assert audit["scenario_count"] == 81
    assert audit["by_phase"] == {"H4-A": 48, "H4-B": 18, "H4-C": 15}
    assert audit["seed_counts"] == {
        "development": 4, "calibration": 10, "confirmation": 10
    }


def test_h4_scenario_artifact_is_a_scaffold_not_a_go_result():
    payload = json.loads(Path(experiment.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
    assert payload["status"] == "PROTOCOL-SCAFFOLD"
    assert "aggregate" not in payload


def test_h4a_smoke_rejects_confirmation_seed_and_unknown_noise():
    presence = dict.fromkeys(experiment.MECHANISMS, True)
    try:
        experiment.run_h4a_smoke(43301, presence, 0.0)
    except ValueError as error:
        assert "development seeds" in str(error)
    else:
        raise AssertionError("confirmation seed entered development smoke")
    try:
        experiment.run_h4a_smoke(43101, presence, 0.75)
    except ValueError as error:
        assert "noise multiplier" in str(error)
    else:
        raise AssertionError("unfrozen noise level entered H4-A")


def test_h4b_smoke_rejects_confirmation_seed_and_unknown_partition():
    try:
        experiment.run_h4b_smoke(43301, "TOD", 0.0)
    except ValueError as error:
        assert "development seeds" in str(error)
    else:
        raise AssertionError("confirmation seed entered H4-B smoke")
    try:
        experiment.run_h4b_smoke(43101, "bad", 0.0)
    except ValueError as error:
        assert "partition" in str(error)
    else:
        raise AssertionError("unknown partition entered H4-B")


def test_h3_discrete_consistent_target_is_zero_noise_only():
    try:
        experiment.h3._assemble(
            43101, "shared", noise_fraction=0.01,
            discrete_consistent_target=True,
        )
    except ValueError as error:
        assert "zero-noise" in str(error)
    else:
        raise AssertionError("noisy predictors entered the exact discrete control")


def test_h4b_protocol_conditions_only_family_and_topology():
    conditioned = experiment.H4B_PROTOCOL["conditioned_dimensions"]
    assert conditioned["family"] == experiment.h3.TRUTH_FAMILY
    assert tuple(conditioned["topology"]) == experiment.h3.TRUTH_TOPOLOGY
    assert conditioned["open_dimension"] == "parameter-sharing partition"

