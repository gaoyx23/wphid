import inspect
import json
from pathlib import Path

from experiments import h4aj_independent_response_source as source
from experiments import paper_h4aj_independent_response_bridge as bridge


def test_independent_source_imports_no_wphid_or_experiment_truth_code():
    text = inspect.getsource(source)
    assert "from experiments" not in text
    assert "from wphid" not in text


def test_independent_packet_contains_observations_only():
    packet = source.generate_packet(43101, "oxygen_damage_power", 0.9)
    assert set(packet) == {"rows", "relative_intervention"}
    assert all("family" not in row and "exponent" not in row for row in packet["rows"])


def test_retained_independent_bridge_matches_protocol():
    payload = json.loads(Path(bridge.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == bridge.PROTOCOL_HASH
    assert payload["aggregate"]["overall_h4_decision"] == "NO-GO"
