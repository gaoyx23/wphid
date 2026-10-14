"""Saved development evidence checks, without any refits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIRST = ROOT / "artifacts/_audits/h3_moment_precision_development"
SECOND = ROOT / "artifacts/_audits/h3_covariance_precision_development"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_eight_pairs_use_identical_packets_and_keep_failures():
    audit = read(FIRST / "audit.json")
    assert audit["complete"] and len(audit["records"]) == 16
    pairs = {}
    for row in audit["records"]:
        pairs.setdefault((row["role"], row["seed"]), []).append(row)
        assert row["signature"] == audit["signature"]
    assert len(pairs) == 8
    assert {s for _, s in pairs} == {42301, 42302, 42303, 42304}
    for rows in pairs.values():
        assert len(rows) == 2
        assert rows[0]["input_packet_sha256"] == rows[1]["input_packet_sha256"]
    assert [s["quantitative_pass"] for s in audit["summary"]] == [3, 3, 3, 3]
    assert audit["new_noise_realizations"] == 0
    assert audit["new_structure_searches"] == 0


def test_frozen_input_sources_have_not_changed():
    protocol = read(FIRST / "protocol.json")
    for path, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    protocol = read(SECOND / "protocol.json")
    for path, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_covariance_development_does_not_overwrite_parent_or_hide_calibration():
    audit = read(SECOND / "audit.json")
    assert audit["complete"] and len(audit["records"]) == 8
    assert audit["additional_noise_calibration_required"]
    assert audit["protocol"]["weight_updates"] == 0
    assert [s["quantitative_pass"] for s in audit["summary"]] == [2, 3]
    assert read(SECOND / "status.json")["state"] == "completed"
    assert not (SECOND / "worker.lock").exists()
