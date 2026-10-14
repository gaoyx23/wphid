import json
from pathlib import Path

from experiments import paper_h4c_noise1_panel as panel


def test_retained_noise1_panel_is_complete_and_does_not_promote_overall_h4():
    payload = json.loads(Path(panel.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == panel.PROTOCOL_HASH
    assert {row["truth_partition"] for row in payload["records"]} == {
        "TOD", "TO|D", "T|O|D"
    }
    assert payload["aggregate"]["correct_usable_outputs"] == 3
    assert payload["aggregate"]["wrong_outputs"] == 0
    assert payload["aggregate"]["panel_decision"] == "DEVELOPMENT-GO"
    assert payload["aggregate"]["overall_h4_decision"] == "NO-GO"
