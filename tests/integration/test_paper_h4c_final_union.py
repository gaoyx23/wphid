import json
from pathlib import Path

from experiments import paper_h4c_final_union as experiment


def test_h4c_generator_contains_all_three_truth_mechanisms():
    _, _, _, states, forcing = experiment._clean("nominal_low_damage", "shared")
    zero = dict.fromkeys(experiment.h1.MECHANISM_CLASSES, 0.0)
    _, _, _, _, base_forcing = experiment.h1._clean("nominal_low_damage", zero)
    assert len(states) == len(forcing) == 3
    assert all(
        not experiment.np.allclose(actual, base)
        for actual, base in zip(forcing, base_forcing)
    )


def test_h4c_smoke_protects_seed_and_noise_splits():
    try:
        experiment.run_smoke(43301, "TOD", 0.0)
    except ValueError as error:
        assert "development seeds" in str(error)
    else:
        raise AssertionError("confirmation seed entered H4-C development")
    try:
        experiment.run_smoke(43101, "TOD", 0.5)
    except ValueError as error:
        assert "noise multiplier" in str(error)
    else:
        raise AssertionError("unfrozen final-union noise entered H4-C")


def test_h4c_partition_tokens_are_safe_for_windows_paths():
    for partition in experiment.h4.REACTION_PARTITIONS:
        token = partition.replace("|", "_")
        assert "|" not in token


def test_h4c_v12_truth_blind_rescue_closes_independent_smoke():
    path = (
        Path("artifacts/paper_h4_complete_mechanism_noise_atlas")
        / "h4c_v12_smoke" / "h4c-T_O_D-noise0p0-seed43101.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
    record = payload["record"]
    assert record["directed_rescue"]["triggered"] is True
    assert record["directed_rescue"]["admitted"] is True
    assert record["directed_rescue"]["propagated_model_count"] == 2
    assert record["structure_equivalence_topology_success"] is True
    assert record["reaction_partition"]["model_choices"] == ["T|O|D"]
    assert record["integrated_endpoint_success"] is True
    assert record["scientific_go_eligible"] is False


def test_h4c_v12_noise1_independent_pilot_passes_all_quantitative_gates():
    path = (
        Path("artifacts/paper_h4_complete_mechanism_noise_atlas")
        / "h4c_v12_noise1_pilot" / "h4c-T_O_D-noise1p0-seed43101.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
    record = payload["record"]
    assert record["directed_rescue"]["admitted"] is True
    assert record["directed_rescue"]["selected"]["topology_name"] == "T"
    assert record["reaction_partition"]["model_choices"] == ["T|O|D"]
    assert record["integrated_endpoint_success"] is True
    assert record["reaction_parameter_gate_all_models"] is True
    assert record["reaction_outer_coefficient_gate_all_models"] is True
    assert record["reaction_action_gate_all_models"] is True


def test_h4c_v12_additional_noise_realizations_repeat_hardest_case_success():
    for seed in (43102, 43103):
        path = (
            Path("artifacts/paper_h4_complete_mechanism_noise_atlas")
            / "h4c_v12_noise1_development"
            / f"h4c-T_O_D-noise1p0-seed{seed}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
        record = payload["record"]
        assert record["directed_rescue"]["admitted"] is True
        assert record["directed_rescue"]["selected"]["topology_name"] == "T"
        assert record["reaction_partition"]["model_choices"] == ["T|O|D"]
        assert record["integrated_endpoint_success"] is True
        assert record["reaction_parameter_gate_all_models"] is True
        assert record["reaction_outer_coefficient_gate_all_models"] is True
        assert record["reaction_action_gate_all_models"] is True
