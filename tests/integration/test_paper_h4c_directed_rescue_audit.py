import json
from pathlib import Path

import numpy as np

from experiments import paper_h4c_directed_rescue_audit as audit


def test_protocol_is_truth_blind_and_searches_complete_directed_grammar():
    assert audit.PROTOCOL["truth_use"] == "post-selection scoring only"
    assert len(audit.PROTOCOL["directed_families"]) == 3
    assert len(audit.PROTOCOL["directed_topologies"]) == 7


def test_retained_rescue_audit_is_diagnostic_only():
    payload = json.loads(Path(audit.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == audit.PROTOCOL_HASH
    assert len(payload["ranking"]) == 21
    assert payload["scientific_go_eligible"] is False
    assert payload["status"] == "DEVELOPMENT-DIAGNOSTIC"
    assert payload["selected"]["topology_name"] == "T"
    assert payload["truth_scoring"]["truth_rank"] == 1
    assert payload["downstream_partition_endpoint"]["selected_partition"] == "T|O|D"
    assert payload["integrated_endpoint_success"] is True


def test_candidate_checkpoint_is_bound_to_data_and_upstream_model(tmp_path):
    bases = (np.eye(2),)
    targets = (np.array([1.0, 2.0]),)
    problem_hash = audit._array_problem_hash(bases, targets)
    changed_hash = audit._array_problem_hash(bases, (np.array([1.0, 2.1]),))
    assert problem_hash != changed_hash
    upstream_hash = audit._upstream_hash({"model": "a"})
    path = tmp_path / "candidate.json"
    audit._write_checkpoint(path, problem_hash, upstream_hash, {"bic": 1.0})
    assert audit._read_checkpoint(path, problem_hash, upstream_hash) == {"bic": 1.0}
    assert audit._read_checkpoint(path, changed_hash, upstream_hash) is None
    assert audit._read_checkpoint(
        path, problem_hash, audit._upstream_hash({"model": "b"})
    ) is None
