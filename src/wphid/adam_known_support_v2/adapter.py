"""Preserve the old adapter; disable support discovery in a known-support task.

This is an explicitly adapted conditional optimizer, not native ADAM-SINDy
structure discovery. It never refines the terminal result with WPHID.
"""

from dataclasses import dataclass

from wphid.final_candidate_v1.adam_adapter import AdamBudget, fit_adam
from wphid.final_candidate_v1.solver import FitBudget


@dataclass(frozen=True)
class KnownSupportBudget:
    starts: FitBudget = FitBudget(seeds=(42, 43, 44))
    epochs_per_start: int = 40000
    learning_rate: float = .01
    scheduler_step: int = 3000
    scheduler_multiplier: float = .5
    maximum_backward_calls: int = 120000

    def adam_budget(self):
        return AdamBudget(starts=self.starts, epochs_per_start=self.epochs_per_start,
                          learning_rate=self.learning_rate, scheduler_step=self.scheduler_step,
                          scheduler_multiplier=self.scheduler_multiplier,
                          maximum_backward_calls=self.maximum_backward_calls,
                          amplitude_threshold=0., sparsity_multiplier=0.)


def fit_known_support(view, candidate, budget=KnownSupportBudget()):
    if not candidate.mechanisms:
        raise ValueError('This task supplies a nonempty active mechanism support')
    result = fit_adam(view, candidate, budget.adam_budget())
    return {**result,
            'version': 'ADAM_SINDY_CONDITIONAL_KNOWN_SUPPORT_V2_20261012',
            'objective_identity': 'sum_equation_mean_weak_residual_squared_no_sparsity_in_supplied_active_support',
            'support_prior': 'Family, active destinations and internal parameter partition are supplied to both programs',
            'effective_sparsity_multiplier': 0., 'amplitude_deletion_threshold': 0.,
            'gamma_effective_in_objective': False,
            'post_fit_profile_refinement': False, 'native_end_to_end_ADAM_SINDy': False,
            'original_adapter_or_old_grades_changed': False}
