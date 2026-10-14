import pytest
from reproducibility.matched_completion_resume_v2 import complete as c

def test_three_lost_cpu_bounds_do_not_claim_total600():
    last={'recorded_utc':'2026-10-13T15:15:40.564197+00:00','states':[{'job_id':j,'job_CPU_seconds_observed':x} for j,x in zip(c.P['pending_ids'],[592.21875,454.8125,142.59375])]}
    a=c.lost_audit(last)
    assert a['observed_lower_CPU_seconds']==1189.625 and a['nominal_allowance_upper_CPU_seconds']==1800
    assert c.P['candidate_total_600_CPU_unchanged_claim'] is False
    last['states'].pop()
    with pytest.raises(ValueError):c.lost_audit(last)

def test_copy_bytes_preserved_and_existing_point_not_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'new')
    p=tmp_path/'old';p.write_bytes(b'{"done":true}\r\n')
    q=c.OUTPUT/'fit'
    c.copy_exact(p,q,c.w.sha(p));assert p.read_bytes()==q.read_bytes()
    with pytest.raises(FileExistsError):c.copy_exact(p,q,c.w.sha(p))
    with pytest.raises(ValueError):c.copy_exact(p,c.OUTPUT/'bad','bad')

def test_runtime_overrides_output_and_administrative_ids_only(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'new')
    monkeypatch.setattr(c.r1,'OUTPUT',c.r1.OUTPUT);monkeypatch.setattr(c.r1,'P',c.r1.P)
    monkeypatch.setattr(c.w,'OUTPUT',c.w.OUTPUT)
    bounds=c.w.ORIGINAL['adam_budget'].copy();c.bind()
    assert c.r1.OUTPUT==c.w.OUTPUT==c.OUTPUT
    assert c.r1.P['potential_inflight_administrative_recoveries']==c.P['pending_ids']
    assert c.w.ORIGINAL['adam_budget']==bounds and bounds['maximum_cpu_seconds']==600
