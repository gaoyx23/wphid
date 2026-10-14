"""Validated data contracts between mechanism identification and inverse health models."""

from __future__ import annotations

from collections.abc import Mapping


STATE_CHANNELS = (
    "T",
    "pO2",
    "pH2O",
    "pProducts",
    "carbon",
    "crack",
    "stress",
)


def validate_mechanism_posterior(packet: Mapping) -> None:
    """Reject unsafe mechanism packets before downstream generative use."""

    required = {
        "schema_version",
        "candidate_law_topology_set",
        "structure_status",
        "parameter_output",
        "parameter_candidate_set",
        "applicability",
    }
    missing = required - set(packet)
    if missing:
        raise ValueError(f"mechanism packet missing keys: {sorted(missing)}")
    candidates = packet["candidate_law_topology_set"]
    if not candidates:
        raise ValueError("candidate law-topology set must not be empty")
    status = packet["structure_status"]
    if status not in {"unique", "candidate_set", "prior_only"}:
        raise ValueError(f"unsupported structure status: {status}")
    if status != "unique" and packet["parameter_output"] is not None:
        raise ValueError(
            "numeric parameter output is forbidden unless structure is unique"
        )
    parameter_set = packet["parameter_candidate_set"]
    if packet["parameter_output"] is not None:
        if not parameter_set or packet["parameter_output"] not in parameter_set:
            raise ValueError(
                "numeric parameter output must belong to its candidate set"
            )
    applicability = packet["applicability"]
    if "nonlinearity_strength_range" not in applicability:
        raise ValueError("applicability must state a nonlinearity strength range")


def validate_inverse_health_contract(contract: Mapping) -> None:
    """Validate the minimum conditioning and output contract for inverse histories."""

    required = {
        "schema_version",
        "conditioning_inputs",
        "generated_targets",
        "physical_constraints",
        "forward_replay_validation",
        "online_role",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"inverse-health contract missing keys: {sorted(missing)}")
    inputs = contract["conditioning_inputs"]
    for key in (
        "damaged_observation",
        "observation_operator",
        "observation_mask",
        "known_load_history",
        "geometry_and_material",
        "mechanism_posterior",
    ):
        if key not in inputs:
            raise ValueError(f"conditioning inputs missing {key}")
    targets = tuple(contract["generated_targets"]["state_channels"])
    if targets != STATE_CHANNELS:
        raise ValueError("generated state channels do not match the seven-field model")
    if contract["online_role"] != "posterior_update_not_de_novo_discovery":
        raise ValueError("single-vehicle online role must remain posterior updating")
    replay = contract["forward_replay_validation"]
    if not replay.get("required", False):
        raise ValueError("forward replay must be mandatory")
    constraints = contract["physical_constraints"]
    if not constraints.get("carbon_nonincreasing", False):
        raise ValueError("carbon irreversibility must be enforced")
    if not constraints.get("crack_nondecreasing", False):
        raise ValueError("crack irreversibility must be enforced")
