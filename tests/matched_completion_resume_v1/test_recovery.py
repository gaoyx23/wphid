import pytest
from reproducibility.matched_completion_resume_v1 import recover as r

def test_copy_preserves_bytes_and_rejects_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUTPUT',tmp_path/'new')
    source=tmp_path/'old.json';source.write_bytes(b'{"completed":true}\r\n')
    target=r.OUTPUT/'fit.json'
    r.copied_exact(source,target,r.original.sha(source))
    assert target.read_bytes()==source.read_bytes()
    with pytest.raises(FileExistsError):r.copied_exact(source,target,r.original.sha(source))

def test_copy_hash_and_root_guards(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUTPUT',tmp_path/'new')
    source=tmp_path/'old';source.write_bytes(b'original')
    with pytest.raises(ValueError,match='identity'):r.copied_exact(source,r.OUTPUT/'fit','wrong')
    with pytest.raises(ValueError,match='escaped'):r.copied_exact(source,tmp_path/'main',r.original.sha(source))

def test_budget_is_explicitly_additional_for_four_administrative_jobs():
    inflight=r.P['potential_inflight_administrative_recoveries']
    jobs=[{'id':name} for name in inflight]+[{'id':f'never-{i}'} for i in range(92)]+[{'id':f'done-{i}'} for i in range(12)]
    audit=r.classify(jobs,{f'done-{i}' for i in range(12)})
    assert audit['pending']==96 and audit['completed_reuses']==12
    assert audit['never_started_conditional_inference']==92
    assert audit['lost_nominal_CPU_seconds_interval']==[0,2400]
    assert r.P['candidate_total_budget_unchanged_claim'] is False

def test_completed_inflight_identity_is_rejected():
    with pytest.raises(ValueError,match='set differs'):r.classify([],set())

def test_recovery_output_guard_and_exclusive_save(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUTPUT',tmp_path/'new')
    r.save(r.OUTPUT/'record',{'x':1})
    with pytest.raises(FileExistsError):r.save(r.OUTPUT/'record',{'x':2})
    with pytest.raises(ValueError,match='escaped'):r.save(tmp_path/'old',{})

def test_runtime_override_only_changes_output(monkeypatch,tmp_path):
    root=r.original.ROOT
    monkeypatch.setattr(r,'OUTPUT',tmp_path/'new')
    monkeypatch.setattr(r.original,'OUTPUT',r.original.OUTPUT)
    r.bind_original()
    assert r.original.OUTPUT==r.OUTPUT
    assert r.original.ROOT==root
    assert r.original.P['maximum_cpu_seconds_per_fit']==600
