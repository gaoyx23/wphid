import json
from pathlib import Path

from wphid.mechanism_interface import (
    validate_inverse_health_contract,
    validate_mechanism_posterior,
)


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r53_mechanism_to_inverse_health_interface/metrics.json"
CONTRACT = ROOT / "artifacts/paper_r53_mechanism_to_inverse_health_interface/conditioning_contract.json"


def test_r53_exports_safe_offline_online_contract():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    prior = payload["offline_mechanism_prior"]
    validate_mechanism_posterior(prior)
    assert prior["weights_calibrated"] is False
    assert prior["parameter_output"] is None
    assert prior["applicability"]["nonlinearity_strength_range"] == [0.5, 1.0]
    aggregate = payload["aggregate"]
    assert aggregate["nested_budget_structure_stable_count"] == 3
    assert aggregate["nested_budget_parameter_stable_count"] == 2
    assert aggregate["online_unique_packet_count"] == 3
    assert aggregate["online_ambiguous_packet_count"] == 1
    assert aggregate["unsafe_parameter_packet_count"] == 0

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    validate_inverse_health_contract(contract)
    assert contract["forward_replay_validation"]["required"] is True
    assert contract["online_role"] == "posterior_update_not_de_novo_discovery"
