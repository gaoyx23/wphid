import json
from pathlib import Path

from experiments import paper_h4as_model_adequacy_guard as guarded


def test_model_adequacy_guard_rejects_nonunit_fixed_axis():
    delta = 0.20
    rows = []
    for block in range(8):
        for axis, elasticity in (("oxygen", 0.30), ("damage", 0.90)):
            for sign in (-1, 1):
                rows.append({
                    "condition": f"block_{block}",
                    "axis": axis,
                    "signed_level": sign,
                    "response": (1.0 + sign * delta) ** elasticity,
                })
    result = guarded.discover({"rows": rows, "relative_intervention": delta})
    assert result["equivalence_guard_selected_family"] == "oxygen_power_damage"
    assert result["selected_family"] == "abstain"
    assert not result["model_adequacy_gate_pass"]


def test_retained_adequacy_guard_development_matches_protocol():
    payload = json.loads(Path(guarded.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == guarded.PROTOCOL_HASH
    assert payload["aggregate"]["prior_h4ar_decision_retained"] == (
        "DEVELOPMENT-NO-GO"
    )
