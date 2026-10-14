import inspect
import json
from pathlib import Path

from experiments import h4ar_independent_finite_volume_response_source as source
from experiments import paper_h4ar_cross_solver_misspecification_audit as audit


def test_finite_volume_source_imports_no_wphid_or_experiment_code():
    text = inspect.getsource(source)
    assert "from experiments" not in text
    assert "from wphid" not in text


def test_finite_volume_packet_contains_observations_only():
    packet = source.generate_packet(44001, "oxygen_damage_power", 0.9)
    assert set(packet) == {"rows", "relative_intervention"}
    assert len(packet["rows"]) == 32
    assert all(
        "family" not in row and "exponent" not in row for row in packet["rows"]
    )


def test_retained_candidate_out_audit_is_no_go():
    payload = json.loads(Path(audit.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == audit.PROTOCOL_HASH
    assert payload["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"
    assert payload["aggregate"]["candidate_out_false_admissions"] > 0
