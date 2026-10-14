"""Temporary software records only; do not load or estimate research data."""

import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reproducibility/precision_development_v2'))
import development_io as io
import run_development as runner


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def setup(monkeypatch,tmp_path):
    root=tmp_path/'software_root'
    output=root/'artifacts/paper_precision_development_v2/software_guard'
    monkeypatch.setattr(io,'ROOT',root)
    monkeypatch.setattr(runner,'ROOT',root)
    monkeypatch.setattr(runner,'OUTPUT',output)
    source=output/'source-freeze.json';manifest=output/'input-manifest.json'
    job={'id':'software_saved','input':{'sha256':'software_input'},'stage':'software_fixture'}
    write(source,{'software_fixture':True})
    write(manifest,{'jobs':[job]})
    record={'job':job,'completed':True,'status':'success','input_sha256':'software_input',
            'source_freeze_sha256':io.sha(source),'input_manifest_sha256':io.sha(manifest)}
    def forbidden(*args,**kwargs):raise AssertionError('A saved/software record must not trigger an estimator')
    monkeypatch.setattr(runner,'fit_factored_candidate',forbidden)
    monkeypatch.setattr(runner,'load_conditions',forbidden)
    return root,output,job,record


def test_writer_rejects_outside_root_and_preserves_completed_file(monkeypatch,tmp_path):
    root,output,_,_=setup(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='escaped'):
        io.write_new(root/'other_directory/result.json',{'software_fixture':True})
    path=output/'new-software-record.json';io.write_new(path,{'original':1})
    before=path.read_bytes()
    with pytest.raises(FileExistsError,match='overwrite'):
        io.write_new(path,{'changed':2})
    assert path.read_bytes()==before


def test_completed_estimate_is_verified_and_skipped_without_any_fit(monkeypatch,tmp_path):
    _,_,job,record=setup(monkeypatch,tmp_path)
    path=runner.output_path(job);write(path,record)
    before=path.read_bytes()
    assert runner.run_job(job)['reused']
    assert path.read_bytes()==before


def test_bound_interrupted_saved_record_is_promoted_without_refitting(monkeypatch,tmp_path):
    _,_,job,record=setup(monkeypatch,tmp_path)
    path=runner.output_path(job);pending=path.with_name(path.name+'.pending-software')
    write(pending,record)
    before=pending.read_bytes()
    result=runner.run_job(job)
    assert result['reused'] and result['saved_pending_promoted']
    assert path.read_bytes()==before


def test_corrupt_completed_identity_blocks_execution(monkeypatch,tmp_path):
    _,_,job,record=setup(monkeypatch,tmp_path)
    record['input_sha256']='different_input';write(runner.output_path(job),record)
    with pytest.raises(ValueError,match='not bound'):
        runner.run_job(job)


def test_job_outside_fixed_manifest_cannot_start_an_estimator(monkeypatch,tmp_path):
    _,_,job,_=setup(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='outside the frozen'):
        runner.run_job({**job,'id':'undeclared_software_case'})
