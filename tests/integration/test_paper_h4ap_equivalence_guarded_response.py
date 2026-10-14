import json
from pathlib import Path

from experiments import paper_h4ap_equivalence_guarded_response as guarded


def test_equivalence_guard_abstains_on_identical_elasticities():
    packet = {
        "relative_intervention": 0.20,
        "rows": [
            {
                "condition": f"block_{block}",
                "axis": axis,
                "signed_level": sign,
                "response": (1.0 + sign * 0.20),
            }
            for block in range(8)
            for axis in ("oxygen", "damage")
            for sign in (-1, 1)
        ],
    }
    assert guarded.discover(packet)["selected_family"] == "abstain"


def test_retained_guarded_development_matches_protocol():
    payload = json.loads(Path(guarded.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == guarded.PROTOCOL_HASH
    assert payload["aggregate"]["prior_h4ao_decision_retained"] == (
        "CONFIRMATION-NO-GO"
    )
