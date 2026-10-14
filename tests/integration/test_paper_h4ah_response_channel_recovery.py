import json
from pathlib import Path

from experiments import paper_h4ah_response_channel_recovery as recovery


def test_discovery_packet_contains_no_truth_metadata():
    packet = recovery.generate_observation_packet(43101, "literal")
    assert set(packet) == {"rows", "relative_intervention"}
    assert all("family" not in row and "exponent" not in row for row in packet["rows"])


def test_retained_response_channel_result_matches_protocol():
    payload = json.loads(Path(recovery.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == recovery.PROTOCOL_HASH
    assert payload["aggregate"]["overall_h4_decision"] == "NO-GO"
