import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('Gaussian_administrative_recovery',ROOT/'reproducibility/wendy_comparison_recovery_v1/workflow.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture():
    jobs=[{'id':f'{s}_{r}'} for s in [9101101,9101102,9101103,9101104,9101105,9101106] for r in module.P['roles']]
    completed=[j['id'] for j in jobs[:12]]
    return jobs,completed

def test_exact_completed_cases_reused_and_only_six_pending_cases_selected():
    jobs,completed=fixture();pending=module.classify(jobs,completed,module.P)
    assert len(pending)==6
    assert not set(j['id'] for j in pending)&set(completed)
    assert set(module.P['administratively_interrupted_jobs'])<=set(j['id'] for j in pending)

def test_completed_case_loss_or_extra_recovery_case_refused():
    jobs,completed=fixture()
    with pytest.raises(ValueError,match='identities'):
        module.classify(jobs,completed[:-1],module.P)
    with pytest.raises(ValueError,match='identities'):
        module.classify(jobs+[{'id':'extra_case'}],completed,module.P)

def test_lost_attempts_are_administrative_with_explicit_cost_allowance():
    jobs,completed=fixture();changed=dict(module.P)
    changed['administratively_interrupted_jobs']=['9101101_shared']
    with pytest.raises(ValueError,match='outside'):
        module.classify(jobs,completed,changed)
    assert module.P['lost_work_cpu_allowance_interval_seconds']==[0,1200]
    assert module.P['no_candidate_total_600_cpu_claim_for_interrupted_jobs']
