"""Independent archive tests; earlier frozen test file stays byte-identical."""
import json
import numpy as np
from reproducibility.coupled_solver_v1.archive_clean_from_noise_replay import recover,FIELD_ORDER,LOAD_ORDER
from reproducibility.coupled_solver_v1.archive_clean_r1 import compare_recovered
from reproducibility.coupled_solver_v1.generator import observe
from reproducibility.coupled_solver_v1.workflow import PROTOCOL


def test_r1_inversion_and_channel_units_on_artificial_fields():
    contract=json.loads(PROTOCOL.read_text(encoding="utf-8"))["observation_contract"]
    fields={k:np.full((5,2,2),v) for k,v in zip(FIELD_ORDER,(.97,100.,6000.,20.,650.,.1,1e7))}
    loads={k:np.full((5,2),v) for k,v in zip(LOAD_ORDER,(1000.,600000.,21000.,500000.))}
    obs,obsl,_,_,_=observe(fields,loads,np.random.default_rng(777),contract)
    recovered,rl=recover(obs,obsl,777,contract)
    assert compare_recovered((recovered,rl,None),(fields,loads,None))<4e-15


def test_r1_previous_defect_is_not_divided_by_later_large_units():
    first=({"carbon":np.array([1.]),"stress":np.array([1e8])},{"pressure":np.array([1e6])},None)
    second=({"carbon":np.array([1.+1e-4]),"stress":np.array([1e8])},{"pressure":np.array([1e6])},None)
    np.testing.assert_allclose(compare_recovered(first,second),1e-4,rtol=1e-10)
