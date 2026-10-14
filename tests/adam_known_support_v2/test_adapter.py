"""Clean, independently constructed artificial qualification problems."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from wphid.adam_known_support_v2 import KnownSupportBudget, fit_known_support
from wphid.final_candidate_v1.problem import WeakCondition, TrainingScale, paper_families
from wphid.final_candidate_v1.solver import Candidate, Mechanism, FitBudget


def artificial(groups):
    rng = np.random.default_rng(20261012)
    states = (rng.uniform(800., 2500., 192), rng.uniform(.25, 1.75, 192), rng.uniform(.02, .8, 192))
    family = next(f for f in paper_families() if f.name == 'arrhenius_oxygen_damage')
    group_parameters = np.asarray([[6000., .8, 1.2], [6800., 1.1, 1.6], [5200., .6, .9]])[:max(groups)+1]
    amplitudes = np.array([1.1, -.8, .6])
    bases = tuple(rng.normal(size=(192, 3)) for _ in range(3))
    targets = tuple(base @ np.array([.002, -.003, .001]) + amplitude*family.values(states, group_parameters[group])
                    for base, amplitude, group in zip(bases, amplitudes, groups))
    condition = WeakCondition('artificial_qualification', bases, targets, states, csr_matrix(np.eye(192)))
    scale = TrainingScale.fit((condition,), (family,))
    candidate = Candidate((Mechanism('reaction', family, groups),))
    return scale.apply((condition,)), candidate, group_parameters, amplitudes


@pytest.mark.parametrize('groups', [(0, 0, 0), (0, 0, 1), (0, 1, 2)])
def test_clean_active_support_and_continuous_parameters_are_recovered(groups):
    import torch
    torch.set_num_threads(1)
    view, candidate, expected, amplitudes = artificial(groups)
    budget = KnownSupportBudget(starts=FitBudget(seeds=(42, 43, 44), maximum_cpu_seconds=120.),
                                epochs_per_start=2000, scheduler_step=1000, maximum_backward_calls=6000)
    result = fit_known_support(view, candidate, budget)
    fit = result['fit']
    recovered = fit.parameters.reshape(-1, 3)
    np.testing.assert_allclose(recovered, expected, rtol=.05, atol=0.)
    actual_amplitudes = np.array([c[-1]/view.scale.norms_for(candidate.mechanisms[0].family)[e]
                                  for e, c in enumerate(fit.coefficients)])
    np.testing.assert_allclose(actual_amplitudes, amplitudes, rtol=.08)
    assert all(abs(a) > 0 for a in actual_amplitudes)
    assert fit.status == 'epoch_budget_completed'
    assert not result['post_fit_profile_refinement']
    assert not result['native_end_to_end_ADAM_SINDy']
    assert result['effective_sparsity_multiplier'] == 0.
    assert result['amplitude_deletion_threshold'] == 0.
