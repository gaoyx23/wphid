import json
from pathlib import Path

from experiments import paper_h4ai_response_augmented_adjudication as adjudication


def test_response_augmented_adjudication_uses_only_retained_paths():
    payload = json.loads(Path(adjudication.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == adjudication.PROTOCOL_HASH
    assert payload["aggregate"]["correct_usable_outputs"] == 4
    assert payload["aggregate"]["wrong_outputs"] == 0
    assert payload["aggregate"]["decision"] == "DEVELOPMENT-GO"
    assert payload["aggregate"]["overall_h4_decision"] == "NO-GO"
