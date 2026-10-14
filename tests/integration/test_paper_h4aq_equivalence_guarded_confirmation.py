import json
from pathlib import Path

from experiments import paper_h4aq_equivalence_guarded_confirmation as confirmation


def test_guarded_confirmation_seeds_are_disjoint_from_development():
    assert set(confirmation.SEEDS).isdisjoint(
        confirmation.guarded.PROTOCOL["seeds"]
    )


def test_retained_guarded_confirmation_matches_locked_protocol():
    payload = json.loads(
        Path(confirmation.DEFAULT_OUTPUT).read_text(encoding="utf-8")
    )
    assert payload["protocol_sha256"] == confirmation.PROTOCOL_HASH
    assert payload["status"] == "CONFIRMATION-LOCKED"
    assert payload["aggregate"]["overall_passive_h4_decision"] == "NO-GO"
