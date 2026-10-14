"""Hand-written software fixtures only; not scientific performance evidence."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from wphid.final_candidate_v1.problem import Family, WeakCondition, TrainingScale, paper_families, validate_split
from wphid.final_candidate_v1.solver import (Candidate, Mechanism, FitBudget, FittedCandidate,
    ScoreConvention, conditional_profile, fit_candidate, partitions, enumerate_candidates,
    predict_frozen, physical_coefficients, search)
from wphid.final_candidate_v1.adam_adapter import AdamBudget, fit_adam
from wphid.final_candidate_v1.observations import reaction_projection, adapt_condition
from wphid.weak.axisymmetric_library import project_axisymmetric_pde_weak
from wphid.final_candidate_v1.pipeline import identify, conditional_pair, change_one_partition


@pytest.fixture(scope="session", autouse=True)
def single_thread_software_runtime():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def fixture(name="train", multiplier=1.):
    states = (np.array([850., 1150., 950., 1350., 1000., 1250.]),
              np.array([.3, 1.2, .8, .5, 1.5, .7]),
              np.array([.04, .3, .6, .1, .5, .2]))
    bases = (np.ones((6, 1)), 2*np.ones((6, 1)), 3*np.ones((6, 1)))
    targets = tuple(multiplier*np.asarray(y) for y in
                    ([2., 3., 1., 5., 2., 4.], [-1., -3., -2., -1., -4., -2.], [1., 4., 2., 3., 5., 1.]))
    projection = csr_matrix(np.array([[1., 0, 0, 0, 0, 0], [0, 1., 0, 0, 0, 0],
                                     [0, 0, 1., 0, 0, 0], [0, 0, 0, 1., 0, 0],
                                     [0, 0, 0, 0, 1., 0], [0, 0, 0, 0, 0, 1.]]))
    return WeakCondition(name, bases, targets, states, projection)


def small_budget(calls=30):
    return FitBudget(seeds=(11,), initial_fractions=(.5,), global_iterations=0,
                     local_iterations=6, maximum_objective_calls=calls, maximum_cpu_seconds=20.)


def test_common_score_includes_null_with_same_floor():
    convention = ScoreConvention.from_targets((np.array([1., 2.]), np.array([3., 4.])))
    equal_rss = (0., 0.)
    assert convention.score(equal_rss, 5)-convention.score(equal_rss, 2) == pytest.approx(3*np.log(4))
    assert np.isfinite(convention.score(equal_rss, 0))


def test_groups_and_counts_do_not_bind_different_mechanisms():
    family = paper_families()[0]
    candidate = Candidate((Mechanism("a", family, (4, 4, None)), Mechanism("b", family, (4, 9, None))))
    assert candidate.mechanisms[0].groups == (0, 0, None)
    assert candidate.mechanisms[1].groups == (0, 1, None)
    assert len(candidate.flat_bounds) == 3
    assert candidate.parameter_count(fixture().bases) == 3+4+3
    with pytest.raises(ValueError):
        Mechanism("bad", family, (True, 0, None))


def test_declared_partition_enumeration():
    rows = tuple(partitions(3))
    assert len(rows) == len(set(rows)) == 14
    assert (0, 0, 0) in rows and (0, 1, 2) in rows and (None, 0, None) in rows
    assert len(enumerate_candidates({"one": paper_families()[:3]}, 3)) == 43


@pytest.mark.parametrize("train,heldout", [(('a',),('bad',)), (('a',),('a',)), (('a','a'),('b',)), ((),('a','b'))])
def test_split_rejects_unknown_duplicate_overlap_and_empty(train, heldout):
    with pytest.raises(ValueError):
        validate_split(('a','b'), train, heldout)


def test_holdout_cannot_change_training_scale_or_enter_optimizer():
    training = fixture()
    holdout = fixture("heldout", 1e8)
    families = paper_families()
    scale = TrainingScale.fit((training,), families)
    before = scale.fingerprint
    holdout_view = scale.apply((holdout,))
    assert scale.fingerprint == before
    assert holdout_view.targets[0][0] == pytest.approx(training.targets[0][0]*scale.target_weights[0]*1e8)
    with pytest.raises(ValueError, match="training"):
        fit_candidate(holdout_view, Candidate(), small_budget())
    with pytest.raises(ValueError, match="training"):
        fit_adam(holdout_view, Candidate((Mechanism("r", families[0], (0, 0, 0)),)),
                 AdamBudget(starts=small_budget(), epochs_per_start=1))


def test_frozen_signed_prediction_and_physical_unit_roundtrip():
    family = paper_families()[0]
    training, heldout = fixture(), fixture("heldout", 1e7)
    scale = TrainingScale.fit((training,), (family,))
    view = scale.apply((training,))
    candidate = Candidate((Mechanism("r", family, (0, None, 0)),))
    coefficients = (np.array([2., -3.]), np.array([-2.]), np.array([1., 4.]))
    fit = FittedCandidate(candidate, np.array([1.1]), coefficients, (1., 1., 1.), 0., 0., 6,
                          scale.fingerprint, "hand_written_fixture", 0, 0., 0., ())
    physical = physical_coefficients(view, fit)
    assert physical[0][-1] < 0 and physical[2][-1] > 0
    predictions = predict_frozen(scale.apply((heldout,)), fit)
    atom = heldout.atom(family, np.array([1.1]))
    for e in range(3):
        expected = heldout.bases[e]@physical[e][:1]
        if e in (0, 2):
            expected = expected+atom*physical[e][-1]
        np.testing.assert_allclose(predictions[e], expected, rtol=1e-14, atol=1e-14)
    wrong_scale = TrainingScale.fit((heldout,), (family,))
    with pytest.raises(ValueError, match="scale"):
        predict_frozen(wrong_scale.apply((heldout,)), fit)
    with pytest.raises(ValueError, match="units"):
        physical_coefficients(wrong_scale.apply((heldout,)), fit)


def test_observation_copy_prevents_caller_mutation():
    c = fixture()
    assert not c.targets[0].flags.writeable and not c.projection.data.flags.writeable
    with pytest.raises(ValueError):
        c.states[0][0] = 1.


@pytest.mark.parametrize("family", paper_families(), ids=lambda f: f.name)
def test_torch_atoms_and_gradients_match_numpy_finite_difference(family):
    import torch
    view = TrainingScale.fit((fixture(),), (family,)).apply((fixture(),))
    p = torch.tensor(family.center, dtype=torch.float64, requires_grad=True)
    atoms = view.torch_builder(family)(p)
    for actual, expected in zip(atoms, view.atom(family, family.center)):
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=2e-14, atol=1e-14)
    gradient = torch.autograd.grad(atoms[0].sum(), p)[0].numpy()
    numeric = []
    for i, c in enumerate(family.center):
        step = max(abs(c), 1.)*1e-6
        plus, minus = np.asarray(family.center).copy(), np.asarray(family.center).copy()
        plus[i] += step
        minus[i] -= step
        numeric.append((view.atom(family, plus)[0].sum()-view.atom(family, minus)[0].sum())/(2*step))
    np.testing.assert_allclose(gradient, numeric, rtol=5e-7, atol=2e-8)


def test_sparse_projection_agrees_with_existing_operator_on_nonuniform_grid():
    t, z, r = (np.array([0., .1, .4, .7, 1.]), np.array([0., .2, .35, .8, 1.]),
               np.array([.1, .2, .4, .7, 1.]))
    field = np.einsum("i,j,k->ijk", 1+t, 2+z*z, 1+r)
    projector = reaction_projection(t, z, r, (5,5,5), (2,2,2))
    matrix, _ = project_axisymmetric_pde_weak(t=t, z=z, r=r, u=field, forcing=0*field,
                     reaction_features=field[..., None], window_shape=(5,5,5), stride_shape=(2,2,2))
    np.testing.assert_allclose(projector@field.ravel(), matrix[:, -1], rtol=2e-14, atol=1e-14)
    c = adapt_condition(name="handwritten_grid", t=t, z=z, r=r,
                        observed_states=(1000+field, .2+.01*field, .1+.01*field),
                        supplied_forcing=(0*field, 0*field, 0*field),
                        window_shape=(5,5,5), stride_shape=(2,2,2))
    assert len(c.bases) == 3 and all(b.shape == (1,3) for b in c.bases)


def test_budget_exhaustion_retains_estimate_and_calls():
    family = paper_families()[0]
    view = TrainingScale.fit((fixture(),), (family,)).apply((fixture(),))
    candidate = Candidate((Mechanism("r", family, (0, 0, 0)),))
    fit = fit_candidate(view, candidate, small_budget(calls=2))
    assert fit.status == "budget_exhausted" and fit.objective_calls == 2
    assert fit.restarts[0]["budget_exhausted"]
    assert np.isfinite(fit.final_score)
    assert family.bounds[0][0] <= fit.parameters[0] <= family.bounds[0][1]


def test_complete_search_uses_null_and_common_final_scores():
    family = paper_families()[0]
    view = TrainingScale.fit((fixture(),), (family,)).apply((fixture(),))
    candidate = Candidate((Mechanism("r", family, (0, 0, 0)),))
    result = search(view, (candidate, Candidate()), maximum_finalists=2, budget=small_budget())
    convention = ScoreConvention.from_targets(view.targets)
    assert result["screened_count"] == 2 and len(result["ranked"]) == 2
    for fit in result["ranked"]:
        assert fit.final_score == pytest.approx(convention.score(fit.equation_rss, fit.nominal_parameter_count))
    assert any(not fit.candidate.mechanisms for fit in result["ranked"])
    assert not result["scientific_qualification_established"]


def test_adam_terminal_score_is_its_own_objective_without_profiling():
    import torch
    torch.set_num_threads(1)
    family = paper_families()[0]
    view = TrainingScale.fit((fixture(),), (family,)).apply((fixture(),))
    candidate = Candidate((Mechanism("r", family, (0, 0, 0)),))
    budget = AdamBudget(starts=small_budget(), epochs_per_start=4,
                        maximum_backward_calls=4, amplitude_threshold=1e6)
    result = fit_adam(view, candidate, budget)
    fit = result["fit"]
    assert result["backward_calls"] == 4 and result["forward_calls"] == 5
    assert all(coef[-1] == 0. for coef in fit.coefficients)
    assert fit.restarts[0]["zero_mechanism_amplitudes"] == 3
    assert fit.training_objective == fit.restarts[0]["terminal_training_objective"]
    assert not result["post_fit_profile_refinement"]
    assert not result["native_end_to_end_ADAM_SINDy"]
    assert fit.status == "epoch_budget_completed"


def test_adam_backward_budget_is_not_reported_as_convergence():
    family = paper_families()[0]
    view = TrainingScale.fit((fixture(),), (family,)).apply((fixture(),))
    candidate = Candidate((Mechanism("r", family, (0, 0, 0)),))
    result = fit_adam(view, candidate, AdamBudget(starts=small_budget(), epochs_per_start=9,
                                                maximum_backward_calls=2))
    assert result["fit"].status == "budget_exhausted"
    assert result["backward_calls"] == 2


def test_integrated_identification_preserves_frozen_holdout_and_unqualified_status():
    family = paper_families()[0]
    candidate = Candidate((Mechanism("r", family, (0, 0, 0)),))
    result = identify((fixture(), fixture("heldout", 1e6)), (Candidate(), candidate),
                      ("train",), ("heldout",), maximum_finalists=2, budget=small_budget())
    assert len(result["ranked_candidates"]) == 2
    assert not result["frozen_heldout"]["refitted"]
    assert result["frozen_heldout"]["scaling_source"] == ["train"]
    assert result["ranked_candidates"][0]["actual_parameter_precision"] is None
    assert not result["reporting"]["parameter_precision_proven_by_this_tier"]
    assert not result["scientific_qualification_established"]


def test_conditional_pair_uses_same_scale_and_preserves_independent_objectives():
    family = paper_families()[0]
    candidate = Candidate((Mechanism("r", family, (0, 0, 0)),))
    result = conditional_pair((fixture(), fixture("heldout")), candidate,
                              ("train",), ("heldout",), wphid_budget=small_budget(),
                              adam_budget=AdamBudget(starts=small_budget(), epochs_per_start=2))
    assert result["wphid"]["training_scale_sha256"] == result["adam_adapter"]["training_scale_sha256"]
    assert not result["wphid_frozen_heldout"]["refitted"] and not result["adam_frozen_heldout"]["refitted"]
    assert not result["truth_used_in_estimators"] and not result["parameter_accuracy_compared"]
    assert not result["adam_adaptation"]["post_fit_profile_refinement"]


def test_sharing_control_changes_only_one_mechanism_binding():
    family = paper_families()[0]
    candidate = Candidate((Mechanism("a", family, (0, 0, 0)), Mechanism("b", family, (0, 0, None))))
    control = change_one_partition(candidate, "a", (0, 1, 2))
    assert control.mechanisms[1] == candidate.mechanisms[1]
    assert control.mechanisms[0].family == candidate.mechanisms[0].family
    assert len(control.flat_bounds)-len(candidate.flat_bounds) == 2
    with pytest.raises(ValueError, match="topology"):
        change_one_partition(candidate, "a", (0, None, 1))
