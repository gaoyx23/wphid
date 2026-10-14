import json
from pathlib import Path

from experiments import paper_h4at_cross_solver_adequacy_confirmation as confirmation


def test_cross_solver_confirmation_seeds_are_disjoint_from_development_and_audit():
    assert set(confirmation.SEEDS).isdisjoint(
        confirmation.guarded.PROTOCOL["seeds"]
    )
    assert set(confirmation.SEEDS).isdisjoint(
        confirmation.audit.PROTOCOL["seeds"]
    )


def test_retained_cross_solver_confirmation_matches_locked_protocol():
    payload = json.loads(
        Path(confirmation.DEFAULT_OUTPUT).read_text(encoding="utf-8")
    )
    assert payload["protocol_sha256"] == confirmation.PROTOCOL_HASH
    assert payload["status"] == "CONFIRMATION-LOCKED"
    assert payload["aggregate"]["overall_passive_h4_decision"] == "NO-GO"
