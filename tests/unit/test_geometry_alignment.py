from dataclasses import replace

import numpy as np
import pytest

from wphid.discovery.geometry_contract import GeometryConvention, geometry_realization_id
from wphid.discovery.observation_pipeline import ObservationPacket, ForwardTemplate, identify_observation_packet
from wphid.discovery.unified import WeakProposal


def convention():
    return GeometryConvention("reference_meridional_arclength","unit_reference_metric",
        "r_ds_dx","radius_face_area")


def geometry():
    s,x=np.arange(3,dtype=float),np.arange(3,dtype=float)
    mesh=dict(volume=np.ones((3,3)),area_s=np.ones((2,3)),area_x=np.ones((3,2)),
              distance_s=np.ones(2),distance_x=np.ones(2))
    return dict(convention=convention(),geometry=mesh,s=s,x=x)


def fixtures():
    cid=convention().convention_id
    p=ObservationPacket(np.array([1.,2.]),np.eye(2),("gas:0","gas:1"),("Pa","Pa"),
                        "port-v1","mission-v1",cid)
    ts=[ForwardTemplate("a","nominal",np.array([1.,2.]),p.labels,p.units,p.operator_id,p.condition_id,cid,"g1"),
        ForwardTemplate("a","shifted",np.array([2.,3.]),p.labels,p.units,p.operator_id,p.condition_id,cid,"g2"),
        ForwardTemplate("b","nominal",np.array([3.,4.]),p.labels,p.units,p.operator_id,p.condition_id,cid,"g1")]
    return p,ts


def identify(p,ts,proposal=None):
    return identify_observation_packet(p,candidate_ids=("a","b"),templates=ts,
        gap_threshold=.01,calibration_id="software-test-only",adequacy_upper_bound=10.,
        weak_adapter=None if proposal is None else lambda _:proposal)


def test_same_rules_allow_multiple_geometry_nuisance_realizations():
    p,ts=fixtures();r=identify(p,ts)
    assert r["selected_nuisance_by_candidate"]["a"]=="nominal"
    assert r["selected_geometry_by_candidate"]["a"]=="g1"
    assert r["geometry_alignment"]=="declared_convention_matched"
    assert r["geometry_declaration_is_not_physical_validation"]


def test_packet_hash_binds_geometry_rules():
    p,_=fixtures()
    assert replace(p,geometry_convention_id="different-rule").observation_id!=p.observation_id


@pytest.mark.parametrize("wrong",[None,"different-rule"])
def test_forward_geometry_rule_missing_or_wrong_is_rejected(wrong):
    p,ts=fixtures()
    ts[0]=replace(ts[0],geometry_convention_id=wrong,geometry_realization_id=None if wrong is None else "g1")
    with pytest.raises(ValueError,match="geometry conventions"):
        identify(p,ts)


@pytest.mark.parametrize("wrong",[None,"different-rule"])
def test_weak_geometry_rule_missing_or_wrong_is_rejected(wrong):
    p,ts=fixtures()
    proposal=WeakProposal(p.observation_id,("b",),True,"software-test-only",wrong,("g1","g2"))
    with pytest.raises(ValueError,match="geometry conventions"):
        identify(p,ts,proposal)


def test_nominal_only_proposal_is_downgraded_not_used_as_complete_profile():
    p,ts=fixtures()
    proposal=WeakProposal(p.observation_id,("b",),True,"software-test-only",p.geometry_convention_id,("g1",))
    r=identify(p,ts,proposal)
    assert r["weak_geometry_profile_complete"] is False
    assert r["weak_proposal_effectively_complete"] is False
    assert r["candidate_set"]==["a"]


def test_complete_geometry_profile_preserves_stage_conflict():
    p,ts=fixtures()
    proposal=WeakProposal(p.observation_id,("b",),True,"software-test-only",p.geometry_convention_id,("g1","g2"))
    r=identify(p,ts,proposal)
    assert r["weak_geometry_profile_complete"]
    assert r["status"]=="stage_conflict"
    assert r["candidate_set"]==["a","b"]


def test_weak_geometry_outside_forward_profile_is_rejected():
    p,ts=fixtures()
    proposal=WeakProposal(p.observation_id,("b",),True,"software-test-only",p.geometry_convention_id,("unknown",))
    with pytest.raises(ValueError,match="outside forward"):
        identify(p,ts,proposal)


def test_realization_hash_changes_with_sizes_not_convention_and_is_order_independent():
    kwargs=geometry();identity=geometry_realization_id(**kwargs)
    kwargs["geometry"]=dict(reversed(list(kwargs["geometry"].items())))
    assert geometry_realization_id(**kwargs)==identity
    kwargs["geometry"]["volume"]=np.full((3,3),2.)
    assert geometry_realization_id(**kwargs)!=identity
    assert kwargs["convention"].convention_id==convention().convention_id


def test_metric_declaration_must_agree_with_numeric_geometry():
    kwargs=geometry();kwargs["geometry"]["meridional_face_metric"]=np.full((2,3),1.1)
    with pytest.raises(ValueError,match="contradicts"):
        geometry_realization_id(**kwargs)
    kwargs=geometry();kwargs["convention"]=replace(convention(),metric_mode="declared_meridional_metric")
    with pytest.raises(ValueError,match="explicitly declared"):
        geometry_realization_id(**kwargs)


def test_legacy_packet_hash_and_status_remain_explicitly_unverified():
    p,ts=fixtures();p=replace(p,geometry_convention_id=None)
    ts=[replace(v,geometry_convention_id=None,geometry_realization_id=None) for v in ts]
    r=identify(p,ts)
    assert r["geometry_alignment"]=="undeclared_legacy_not_verified"
    assert r["geometry_convention_id"] is None
