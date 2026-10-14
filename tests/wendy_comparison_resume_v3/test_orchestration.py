from pathlib import Path
import pytest
from reproducibility.wendy_comparison_resume_v3 import complete as c

def test_checkpoint_rejects_recycled_process():
    class Process:
        pid=19
        def create_time(self):return 3.
    with pytest.raises(ValueError):c.cpu_snapshot({'pid':19,'process_create_time':2.},Process())

def test_predeclared_thresholds_and_same_packets_remain_bound():
    c.verify_declarations()
    p=c.f.read(c.f.PROTO)
    assert p['entire_fit_cpu_seconds']==600 and p['starts']==[.25,.5,.75]
    assert p['fresh_blocks']==list(range(9101301,9101319))
    clarification=c.f.read(c.ROOT/'reproducibility/wendy_comparison_v2/independent-directional-family-clarification-r1-20261013.json')
    assert len(clarification['pairwise_program_pairs'])==3

def test_outputs_do_not_overwrite_a_completed_receipt(tmp_path):
    path=tmp_path/'saved.json';path.write_bytes(b'original')
    with pytest.raises(FileExistsError):c.save(path,{'changed':True})
    assert path.read_bytes()==b'original'
