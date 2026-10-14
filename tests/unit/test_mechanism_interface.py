import pytest

from wphid.mechanism_interface import (
    STATE_CHANNELS,
    validate_inverse_health_contract,
    validate_mechanism_posterior,
)


def test_mechanism_packet_forbids_parameters_under_structural_ambiguity():
    packet = {
        "schema_version": "1.0",
        "candidate_law_topology_set": [
            ["Forchheimer", "isotropic"],
            ["Forchheimer", "directional"],
        ],
        "structure_status": "candidate_set",
        "parameter_output": 0.5,
        "parameter_candidate_set": [0.5, 1.0],
        "applicability": {"nonlinearity_strength_range": [0.5, 1.0]},
    }
    with pytest.raises(ValueError, match="forbidden"):
        validate_mechanism_posterior(packet)


def test_inverse_contract_requires_forward_replay_and_seven_fields():
    contract = {
        "schema_version": "1.0",
        "conditioning_inputs": {
            "damaged_observation": {},
            "observation_operator": {},
            "observation_mask": {},
            "known_load_history": {},
            "geometry_and_material": {},
            "mechanism_posterior": {},
        },
        "generated_targets": {"state_channels": STATE_CHANNELS},
        "physical_constraints": {
            "carbon_nonincreasing": True,
            "crack_nondecreasing": True,
        },
        "forward_replay_validation": {"required": True},
        "online_role": "posterior_update_not_de_novo_discovery",
    }
    validate_inverse_health_contract(contract)
