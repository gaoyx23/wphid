"""Artificial six-row software fixtures; no scientific recovery endpoint."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import eye

from wphid.final_candidate_v1.problem import WeakCondition, TrainingScale, paper_families
from wphid.final_candidate_v1.solver import (
    Candidate, Mechanism, FitBudget, ScoreConvention, conditional_profile,
    enumerate_candidates, fit_candidate, predict_frozen)
from wphid.precision_development_v2.factorization import (
    factorize, prepare_profile, profile_components, component_profile)
from wphid.precision_development_v2.solver import allocate_calls, fit_factored_candidate
import wphid.precision_development_v2.solver as component_solver


def observation(name="software_train", rank_deficient=False, exact_base=False):
    states = (np.array([850., 1150., 950., 1350., 1000., 1250.]),
              np.array([.3, 1.2, .8, .5, 1.5, .7]),
              np.array([.04, .3, .6, .1, .5, .2]))
    bases = tuple(np.column_stack([np.ones(6), np.ones(6)]) if rank_deficient
                  else np.ones((6, 1)) for _ in range(3))
    targets = tuple(np.array(y, dtype=float) for y in
                    ([2,3,1,5,2,4], [-1,-3,-2,-1,-4,-2], [1,4,2,3,5,1]))
    if exact_base:
        targets = tuple(np.full(6, float(e+1)) for e in range(3))
    return WeakCondition(name, bases, targets, states, eye(6, format="csr"))


def view(rank_deficient=False, exact_base=False):
    obs = observation(rank_deficient=rank_deficient, exact_base=exact_base)
    return TrainingScale.fit((obs,), paper_families()).apply((obs,))


def budget(calls=60):
    return FitBudget(seeds=(11011,), initial_fractions=(.25,), global_iterations=0,
                     local_iterations=6, maximum_objective_calls=calls, maximum_cpu_seconds=20.)


def reaction(groups):
    return Candidate((Mechanism("reaction", paper_families()[6], groups),))


@pytest.mark.parametrize("rank_deficient,exact_base", [(False,False),(True,False),(False,True)])
def test_all_43_candidates_preserve_original_profile_score_and_coefficient_layout(rank_deficient,exact_base):
    training = view(rank_deficient,exact_base)
    candidates = enumerate_candidates({"reaction":paper_families()[6:]},3)
    assert len(candidates)==43
    convention = ScoreConvention.from_targets(training.targets)
    for candidate in candidates:
        bounds=np.asarray(candidate.flat_bounds).reshape(-1,2)
        for fraction in (.23,.57):
            point=bounds[:,0]+fraction*(bounds[:,1]-bounds[:,0])
            original_coefficients,original_rss=conditional_profile(training,candidate,point)
            profile=profile_components(prepare_profile(training,candidate),point)
            np.testing.assert_allclose(profile.equation_rss,original_rss,rtol=2e-13,atol=1e-26)
            for actual,expected in zip(profile.coefficients,original_coefficients):
                np.testing.assert_array_equal(actual,expected)
            assert profile.training_objective==pytest.approx(convention.objective(original_rss),abs=2e-11)
            assert sum(profile.component_objectives)+profile.inactive_objective==pytest.approx(profile.training_objective,abs=2e-11)
            assert profile.final_score==pytest.approx(convention.score(original_rss,candidate.parameter_count(training.bases)),abs=2e-11)


def test_transitive_cross_mechanism_coupling_is_not_split():
    families=paper_families()
    candidate=Candidate((Mechanism("damage",families[0],(0,None,None)),
                         Mechanism("oxygen",families[3],(0,0,None)),
                         Mechanism("reaction",families[6],(None,0,1))))
    graph=factorize(candidate,3)
    assert [c.equations for c in graph.components]==[(0,1),(2,)]
    assert [len(c.parameter_indices) for c in graph.components]==[5,3]
    profile=profile_components(prepare_profile(view(),candidate),candidate.center)
    _,original=conditional_profile(view(),candidate,candidate.center)
    np.testing.assert_array_equal(profile.equation_rss,original)


def test_equal_group_labels_in_different_mechanisms_do_not_bind_disjoint_equations():
    candidate=Candidate((Mechanism("a",paper_families()[0],(0,None,None)),
                         Mechanism("b",paper_families()[3],(None,0,None))))
    graph=factorize(candidate,3)
    assert [c.equations for c in graph.components]==[(0,),(1,)]
    assert graph.inactive_equations==(2,)


def test_noncontiguous_global_parameters_keep_original_mechanism_column_order():
    candidate=Candidate((Mechanism("a",paper_families()[0],(0,1,None)),
                         Mechanism("b",paper_families()[6],(0,None,1))))
    graph=factorize(candidate,3)
    assert graph.components[0].parameter_indices==(0,2,3,4)
    training=view();point=candidate.center.copy();point[0]=.8;point[1]=1.7
    profile=profile_components(prepare_profile(training,candidate),point)
    coefficients,rss=conditional_profile(training,candidate,point)
    np.testing.assert_array_equal(profile.equation_rss,rss)
    for actual,expected in zip(profile.coefficients,coefficients):
        np.testing.assert_array_equal(actual,expected)


def test_changing_one_component_leaves_other_equations_unchanged():
    training=view();candidate=reaction((0,0,1));problem=prepare_profile(training,candidate)
    original=profile_components(problem,candidate.center)
    changed=candidate.center.copy();changed[3:]=[7000.,.8,1.8]
    modified=profile_components(problem,changed)
    np.testing.assert_array_equal(original.equation_rss[:2],modified.equation_rss[:2])
    for e in (0,1):np.testing.assert_array_equal(original.coefficients[e],modified.coefficients[e])


def test_rejects_a_component_from_another_family_and_invalid_parameter_vector():
    candidate=reaction((0,1,2));problem=prepare_profile(view(),candidate)
    foreign=factorize(Candidate((Mechanism("reaction",paper_families()[7],(0,1,2)),)),3).components[0]
    with pytest.raises(ValueError,match="different candidate"):
        component_profile(problem,foreign,np.array([8000.,1.]))
    with pytest.raises(ValueError,match="vector"):
        component_profile(problem,problem.factorization.components[0],np.array([np.nan,1.,1.]))


def test_holdout_and_destination_mismatch_are_rejected():
    training=view();held=training.scale.apply((observation("software_holdout"),))
    with pytest.raises(ValueError,match="training"):
        fit_factored_candidate(held,reaction((0,1,2)),budget())
    with pytest.raises(ValueError,match="training"):
        prepare_profile(held,reaction((0,1,2)))
    with pytest.raises(ValueError,match="destination"):
        factorize(reaction((0,1,2)),2)


@pytest.mark.parametrize("calls",[1,2,4,10])
def test_all_components_share_one_call_cap_and_retain_a_feasible_baseline(calls):
    training=view();candidate=reaction((0,1,2))
    result=fit_factored_candidate(training,candidate,budget(calls))
    assert result.fit.objective_calls<=calls
    assert result.fit.objective_calls==1+sum(r["objective_calls"] for r in result.audit["component_records"])
    assert result.audit["call_quota_sum"]==calls-1
    assert result.audit["CPU_quota_sum"]<=20.
    assert all(r["objective_calls"]<=r["maximum_objective_calls"] for r in result.audit["component_records"])
    assert np.isfinite(result.fit.final_score)
    assert result.audit["final_coefficient_refits"]==0
    if calls==1:
        bounds=np.asarray(candidate.flat_bounds)
        np.testing.assert_array_equal(result.fit.parameters,bounds[:,0]+.25*(bounds[:,1]-bounds[:,0]))
    coefs,rss=conditional_profile(training,candidate,result.fit.parameters)
    np.testing.assert_array_equal(result.fit.equation_rss,rss)
    for actual,expected in zip(result.fit.coefficients,coefs):np.testing.assert_array_equal(actual,expected)
    prediction=predict_frozen(training,result.fit)
    for y,p,r,w in zip(training.targets,prediction,rss,training.scale.target_weights):
        residual=y-p*w  # predict_frozen returns physical units; targets are scaled.
        assert float(residual@residual)==pytest.approx(r,rel=2e-12,abs=1e-26)


def test_program_cpu_cap_stops_later_blocks_and_records_between_call_overshoot(monkeypatch):
    clock=[0.];original=component_solver.component_profile
    monkeypatch.setattr(component_solver,"process_time",lambda:clock[0])
    def timed_profile(*args):
        answer=original(*args);clock[0]+=1.;return answer
    monkeypatch.setattr(component_solver,"component_profile",timed_profile)
    result=fit_factored_candidate(view(),reaction((0,1,2)),replace(budget(300),maximum_cpu_seconds=4.))
    assert result.audit["global_CPU_cap_reached"]
    assert result.fit.cpu_seconds==4.
    assert result.audit["component_records"][-1]["objective_calls"]==0
    assert result.fit.objective_calls==5
    assert result.fit.status=="budget_exhausted"


@pytest.mark.parametrize("candidate",[Candidate(),reaction((0,0,0)),
    Candidate((Mechanism("a",paper_families()[0],(0,0,None)),
               Mechanism("b",paper_families()[6],(None,0,0))))])
def test_null_or_connected_candidates_delegate_without_changing_estimates(candidate):
    training=view();original=fit_candidate(training,candidate,budget())
    result=fit_factored_candidate(training,candidate,budget())
    assert result.audit["mode"]=="immutable_V1_delegate"
    np.testing.assert_array_equal(result.fit.parameters,original.parameters)
    np.testing.assert_array_equal(result.fit.equation_rss,original.equation_rss)
    assert result.fit.final_score==original.final_score
    assert result.fit.objective_calls==original.objective_calls


def test_baseline_is_evaluated_once_and_assembly_does_not_refit(monkeypatch):
    baseline_calls=[];original=component_solver.conditional_profile
    def counted(*args):baseline_calls.append(1);return original(*args)
    monkeypatch.setattr(component_solver,"conditional_profile",counted)
    result=fit_factored_candidate(view(),reaction((0,1,2)),budget())
    assert len(baseline_calls)==1
    assert result.audit["final_coefficient_refits"]==0
    assert all(r["best_objective"]<=r["baseline_objective"] for r in result.audit["component_records"])


def test_call_quota_allocation_has_deterministic_ties_and_preserves_total():
    assert allocate_calls(2,(3,3,3))==(1,1,0)
    assert allocate_calls(10,(1,3))==(3,7)
    assert allocate_calls(0,(1,2))==(0,0)
