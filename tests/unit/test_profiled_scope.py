import pytest
from wphid.discovery.profiled_scope import qualify_profiled_evidence


def call(**changes):
    args=dict(decision=dict(candidate_set=["a"],scores={"a":{},"b":{}}),
        candidate_universe=["a","b"],calibration_id="ref",calibrated_prefactor_points=[(1.,1.)])
    args.update(changes)
    return qualify_profiled_evidence(**args)


def test_unknown_parameters_and_support_cannot_inherit_reference_pruning():
    result=call()
    assert result["candidate_set"]==["a","b"]
    assert result["conditional_candidate_set"]==["a"]
    assert len(result["scope_failures"])==2
    assert not result["weak_proposal_complete"]
    assert not result["hard_pruning_authorized"]


def test_known_reference_assumptions_leave_only_conditional_evidence():
    result=call(requested_prefactors=(1,1),requested_reaction_support="oxygen_and_steam")
    assert result["applicability_qualified"]
    assert result["candidate_set"]==["a"]
    assert result["parameter_output"] is None
    assert not result["uniform_unknown_parameter_recovery"]


def test_out_of_scope_empty_set_not_interpreted_as_missing_physics():
    result=call(decision=dict(candidate_set=[],scores={"a":{},"b":{}}),requested_prefactors=(.8,1),requested_reaction_support="oxygen_and_steam")
    assert result["candidate_set"]==["a","b"]
    assert result["conditional_candidate_set"]==[]
    assert result["status"]=="advisory_only_scope_unvalidated"


def test_incomplete_library_rejected():
    with pytest.raises(ValueError):call(decision=dict(candidate_set=["a"],scores={"a":{}}))


def test_parameter_reference_points_not_auto_interpolated():
    result=call(calibrated_prefactor_points=[(.8,1),(1,1)],requested_prefactors=(.9,1),requested_reaction_support="oxygen_and_steam")
    assert not result["applicability_qualified"]
