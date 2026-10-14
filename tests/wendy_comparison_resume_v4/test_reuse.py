from pathlib import Path
import pytest
from reproducibility.wendy_comparison_resume_v4 import complete as c

def test_pause_partition_requires_exact_unstarted_jobs():
    jobs=[{'id':str(i),'input_sha256':str(i)} for i in range(54)]
    completed={str(i):'saved' for i in range(20)}
    assert c.pending_jobs(jobs,completed,jobs[20:])==jobs[20:]
    with pytest.raises(ValueError):c.pending_jobs(jobs,completed,jobs[21:]+jobs[:1])

def test_existing_endpoint_never_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    source=tmp_path/'old.json';target=tmp_path/'new.json'
    source.write_bytes(b'old saved');target.write_bytes(b'user saved')
    with pytest.raises(FileExistsError):c.copy_exact(source,target,c.f.sha(source))
    assert target.read_bytes()==b'user saved'

def test_copy_checks_source_bytes_and_is_independent(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    source=tmp_path/'old.json';target=tmp_path/'copies/new.json';source.write_bytes(b'exact')
    with pytest.raises(ValueError):c.copy_exact(source,target,'changed')
    c.copy_exact(source,target,c.f.sha(source));source.write_bytes(b'other')
    assert target.read_bytes()==b'exact'
