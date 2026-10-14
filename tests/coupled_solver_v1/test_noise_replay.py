"""Archival noise inversion on artificial arrays, no physical integration."""

import json
import numpy as np
from reproducibility.coupled_solver_v1.archive_clean_from_noise_replay import recover,FIELD_ORDER,LOAD_ORDER
from reproducibility.coupled_solver_v1.generator import observe
from reproducibility.coupled_solver_v1.workflow import PROTOCOL


def test_all_fields_loads_and_fixed_initial_recover_original_artificial_arrays():
    contract=json.loads(PROTOCOL.read_text(encoding="utf-8"))["observation_contract"]
    shape=(5,2,2);fields={k:np.full(shape,value) for k,value in zip(FIELD_ORDER,(.97,100.,6000.,20.,650.,.1,1e7))}
    loads={k:np.full((5,2),v) for k,v in zip(LOAD_ORDER,(1000.,600000.,21000.,500000.))}
    obs,obsl,_,_,_=observe(fields,loads,np.random.default_rng(777),contract)
    recovered,rl=recover(obs,obsl,777,contract)
    for k in fields:np.testing.assert_allclose(recovered[k],fields[k],rtol=4e-15,atol=1e-15)
    for k in loads:np.testing.assert_allclose(rl[k],loads[k],rtol=4e-15)
