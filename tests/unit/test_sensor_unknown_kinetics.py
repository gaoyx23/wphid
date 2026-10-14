import json
import numpy as np
from experiments import run_sensor_local_unknown_kinetics as kinetics


def fields():
    t=np.linspace(0,300,61)
    shape=(61,2,3)
    return t,dict(T=np.broadcast_to(np.linspace(1000,1900,61)[:,None,None],shape),
        pO2=np.ones(shape)*9000,pH2O=np.ones(shape)*8e5,
        carbon=np.broadcast_to(np.linspace(1,.92,61)[:,None,None],shape))


def test_analytic_rate_jacobian_matches_finite_difference():
    _,f=fields();p=(kinetics.PROBLEMS["chemistry"]["lower"]+kinetics.PROBLEMS["chemistry"]["upper"])/2
    rate,jac=kinetics.rate_and_jacobian(p,f)
    for j in range(7):
        plus=p.copy();minus=p.copy();plus[j]+=1e-5;minus[j]-=1e-5
        numeric=(kinetics.rate_and_jacobian(plus,f)[0]-kinetics.rate_and_jacobian(minus,f)[0])/2e-5
        np.testing.assert_allclose(jac[...,j],numeric,rtol=1e-7,atol=1e-16)


def test_batched_temporal_projection_matches_public_sensor_operator():
    t,f=fields();p=(kinetics.PROBLEMS["chemistry"]["lower"]+kinetics.PROBLEMS["chemistry"]["upper"])/2
    rate,_=kinetics.rate_and_jacobian(p,f)
    predicted,_=kinetics.weak_prediction(p,dict(fields=f,kernel=kinetics.temporal_kernel(t)))
    matrix,_=kinetics.project_sensor_time_weak(t=t,u=f["carbon"],features=rate[...,None],window=31,stride=15)
    np.testing.assert_allclose(predicted,matrix[:,0],rtol=1e-13,atol=1e-17)


def test_saved_unknown_parameter_stage_preserves_scope_and_sealed_holdout():
    result=json.loads((kinetics.OUT/"audit.json").read_text())
    assert result["complete"] and result["optimizer_calls"]==6
    assert result["new_simulations"]==result["new_random_realizations"]==result["random_stream_replays"]==0
    assert result["conditional_on_two_reaction_support"]
    assert result["heldout_not_used_for_parameter_selection"]
    assert result["parameter_confidence_intervals"] is None
    for row in result["records"].values():
        assert len(row["fits"])==3
        assert row["selected"]["objective_norm"]==min(fit["objective_norm"] for fit in row["fits"])
        assert len(row["selected"]["physical_parameters"])==7


def test_task_transfer_initialization_is_blind_to_held_task():
    result=json.loads((kinetics.OUT/"task_transfer/audit.json").read_text())
    assert result["complete"] and result["new_optimizer_calls"]==3
    assert result["parameter_confidence_intervals"] is None
    for held,row in result["records"].items():
        assert held not in row["training"]
        assert len(row["training"])==2
