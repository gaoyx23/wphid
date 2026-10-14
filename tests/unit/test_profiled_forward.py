import numpy as np
import pytest
from wphid.discovery.profiled_forward import build_forward_template,score_forward,reconcile_same_packet,inverse_noise


def test_simplex_profile_recovers_interior_and_anchor_parameters():
    reference=np.ones(6);oxygen=reference+np.arange(6);steam=reference+np.arange(6)[::-1]
    template=build_forward_template(reference=reference,oxygen_anchor=oxygen,steam_anchor=steam,scale=np.ones(6),modes=np.ones((6,1)))
    for weights in ([0,0],[1,0],[0,1],[.5,.25]):
        data=reference+(oxygen-reference)*weights[0]+(steam-reference)*weights[1]
        result=score_forward(data,template)
        np.testing.assert_allclose(result["triangle_weights"],weights,atol=1.e-12)
        assert result["weighted_squared_residual"]<1.e-20


def test_noise_inverse_matches_dense_covariance():
    rng=np.random.default_rng(3);modes=rng.normal(size=(12,2));values=rng.normal(size=(12,3))
    np.testing.assert_allclose(inverse_noise(values,modes),np.linalg.solve(np.eye(12)+modes@modes.T,values),atol=1.e-13)


def test_unqualified_weak_result_cannot_hard_prune_or_expand_forward_set():
    result=reconcile_same_packet(universe=["a","b"],forward_set=["a"],weak_set=[],forward_observation_id="packet",weak_observation_id="packet")
    assert result["candidate_set"]==["a"] and result["weak_evidence_advisory_only"]
    with pytest.raises(ValueError):reconcile_same_packet(universe=["a","b"],forward_set=["a"],weak_set=[],forward_observation_id="one",weak_observation_id="two")


def test_qualified_conflict_expands_and_withholds_parameters():
    result=reconcile_same_packet(universe=["a","b"],forward_set=["a"],weak_set=["b"],forward_observation_id="packet",weak_observation_id="packet",weak_applicability_qualified=True)
    assert result["candidate_set"]==["a","b"] and result["status"]=="stage_conflict"
    assert result["parameter_output"] is None
