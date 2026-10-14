import hashlib
import json
import pytest
from reproducibility.matched_completion_v1 import workflow as w

def test_hash_reuse_rejects_changed_completed_input(tmp_path, monkeypatch):
    monkeypatch.setattr(w,'ROOT',tmp_path)
    p=tmp_path/'input.npz'
    p.write_bytes(b'original')
    bound={'input.npz':hashlib.sha256(b'original').hexdigest()}
    w.verify_hashes(bound)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen identity changed'):
        w.verify_hashes(bound)

def test_computability_requires_each_role_and_no_thrown_failure():
    rows=[{'role':role,'assessable':i<5,'success':True} for role in w.P['roles'] for i in range(6)]
    assert w.computability_gate(rows)['passed']
    rows[0]['assessable']=False
    assert not w.computability_gate(rows)['passed']
    rows[0]['assessable']=True
    rows[0]['success']=False
    assert not w.computability_gate(rows)['passed']

def test_precision_not_computability_gate():
    rows=[{'role':role,'assessable':True,'success':True,'precision_pass':False} for role in w.P['roles'] for _ in range(6)]
    assert w.computability_gate(rows)['passed']

def test_development_reuses_wphid_and_fresh_pairs_both():
    p={'seed':1,'role':'shared','file':'unused','sha256':'unused'}
    assert [j['program'] for j in w.jobs('development',[p])]==['adam']
    assert [j['program'] for j in w.jobs('fresh',[p])]==['wphid','adam']

def test_exclusive_output_and_root_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'OUTPUT',tmp_path/'independent')
    p=w.OUTPUT/'record.json'
    w.save(p,{'completed':True})
    with pytest.raises(FileExistsError):w.save(p,{'completed':False})
    with pytest.raises(ValueError,match='escaped'):w.save(tmp_path/'main.json',{})

def test_completed_failure_is_identity_checked_not_retried(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'OUTPUT',tmp_path)
    w.save(tmp_path/'source-freeze.json',{})
    job={'id':'one','stage':'development','input':{'sha256':'abc'}}
    record={'job':job,'input_sha256':'abc','source_freeze_sha256':w.sha(tmp_path/'source-freeze.json'),'completed':True,'status':'failed'}
    w.save(w.job_path(job),record)
    assert w.run_job(job)=={'id':'one','reused':True,'status':'failed'}
    record['input_sha256']='changed'
    with pytest.raises(ValueError):w.validate_record(record,job)
