import hashlib,json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/"artifacts/_audits/envelope_unified_stage/audit.json"


@pytest.fixture
def saved():
    if not STAGE.exists():pytest.skip("unified stage unavailable")
    report=json.loads(STAGE.read_text(encoding="utf-8"))
    if not report["complete"]:pytest.skip("unified stage still running")
    return report


def test_conditional_weak_failure_does_not_disappear_from_audit(saved):
    weak=json.loads((ROOT/"artifacts/_audits/conditional_profile_stage/model.json").read_text(encoding="utf-8"))
    assert not weak["development_gate_met"]
    assert saved["protocol"]["weak_stage_advisory"]
    assert not saved["protocol"]["threshold_search"]
    assert saved["protocol"]["surrogate_rms_budget"]==.25


def test_frozen_model_identity_and_confirmation_counts_match(saved):
    if not saved["confirmation_run"]:pytest.skip("development gate failed")
    assert saved["frozen_model_sha256"]==hashlib.sha256((STAGE.parent/"frozen_model.json").read_bytes()).hexdigest()
    for case,truth,_,_ in saved["protocol"]["confirmation_cases"]:
        rows=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((STAGE.parent/"draws"/case).glob("*.json"))]
        assert len(rows)==128
        assert sum(truth in r["decision"]["candidate_set"] for r in rows)==saved["confirmation"][case]["truth_retained"]
        for row in rows:
            assert row["model_sha256"]==saved["frozen_model_sha256"]
            assert row["decision"]["parameter_output"] is None
            assert row["decision"]["canonical_candidate_set"]==[saved["canonical_candidate_ids"][n] for n in row["decision"]["candidate_set"]]


def test_surrogate_qualification_is_not_continuous_domain_certification(saved):
    if not saved["confirmation_run"]:pytest.skip("development gate failed")
    assert len(saved["surrogate_qualification"])==4
    for result in saved["surrogate_qualification"].values():
        assert not result["continuous_domain_certified"]
        assert result["target_met"]==(result["normalized_bias_rms"]<=.25)
