"""Conditional common-weak-input ADAM-SINDy adaptation, not native discovery.

Reference: official commit 1b301baa6e7b4a90a255466fd0a2cbf3d10bb200,
chemical notebook blob c7677a0dad89a9f10a0bffe3aa4e792862ddeca0.
This is a new implementation of the stated joint-minimization branch. Weak
inputs, bounded scaled nonlinear parameters, mandatory unpenalized base terms,
entrywise amplitude penalty and training-scale threshold are explicit changes.
No WPHID refinement is applied to the returned terminal Adam estimate.
"""

from dataclasses import dataclass
import time

import numpy as np

from .solver import Candidate, FittedCandidate, FitBudget, ScoreConvention, conditional_profile


@dataclass(frozen=True)
class AdamBudget:
    starts: FitBudget = FitBudget(seeds=(42, 43, 44))
    epochs_per_start: int = 40000
    learning_rate: float = .01
    scheduler_step: int = 3000
    scheduler_multiplier: float = .5
    amplitude_threshold: float = .005
    sparsity_multiplier: float = 1.
    maximum_backward_calls: int = 120000

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in
               (self.epochs_per_start, self.scheduler_step, self.maximum_backward_calls)):
            raise ValueError("Invalid Adam iteration budget")
        if (not np.isfinite([self.learning_rate, self.scheduler_multiplier,
                             self.amplitude_threshold, self.sparsity_multiplier]).all()
                or self.learning_rate <= 0 or not 0 < self.scheduler_multiplier <= 1
                or self.amplitude_threshold < 0 or self.sparsity_multiplier < 0):
            raise ValueError("Invalid Adam loss or threshold settings")


def fit_adam(view, candidate: Candidate, budget=AdamBudget()):
    import torch
    if not candidate.mechanisms:
        raise ValueError("Conditional Adam comparator requires declared nonempty mechanisms")
    if tuple(c.name for c in view.conditions) != view.scale.condition_names:
        raise ValueError("Adam must fit only frozen training conditions")
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    bases = tuple(torch.tensor(b.copy(), dtype=torch.float64) for b in view.bases)
    targets = tuple(torch.tensor(y.copy(), dtype=torch.float64) for y in view.targets)
    bounds = np.asarray(candidate.flat_bounds)
    lower = torch.tensor(bounds[:, 0].copy(), dtype=torch.float64)
    span = torch.tensor((bounds[:, 1]-bounds[:, 0]).copy(), dtype=torch.float64)
    count = candidate.parameter_count(view.bases)
    builders = tuple(view.torch_builder(m.family) for m in candidate.mechanisms)
    layouts, offset = [], 0
    for m in candidate.mechanisms:
        width = len(m.family.bounds)
        layouts.append(tuple(slice(offset+g*width, offset+(g+1)*width) for g in range(m.group_count)))
        offset += width*m.group_count
    amplitude_indices = tuple(tuple(range(b.shape[1], b.shape[1]+sum(m.groups[e] is not None
                                  for m in candidate.mechanisms))) for e, b in enumerate(bases))
    best, records, backward_calls, forward_calls = None, [], 0, 0

    for seed, fraction in zip(budget.starts.seeds, budget.starts.initial_fractions):
        if time.process_time()-started_cpu >= budget.starts.maximum_cpu_seconds:
            break
        p = torch.nn.Parameter(torch.full_like(lower, fraction))
        initial_values = (lower+p*span).detach().numpy()
        initial_coefficients, _ = conditional_profile(view, candidate, initial_values)
        coefficients = tuple(torch.nn.Parameter(torch.tensor(c.copy(), dtype=torch.float64))
                             for c in initial_coefficients)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        gamma = tuple(torch.nn.Parameter(torch.randn(len(i), generator=generator, dtype=torch.float64))
                      for i in amplitude_indices)
        optimizer = torch.optim.Adam([p, *coefficients], lr=budget.learning_rate, betas=(.9, .99), eps=1e-15)
        weight_optimizer = torch.optim.Adam(gamma, lr=budget.learning_rate, betas=(.9, .99), eps=1e-15)
        schedulers = tuple(torch.optim.lr_scheduler.StepLR(o, step_size=budget.scheduler_step,
                                                          gamma=budget.scheduler_multiplier)
                           for o in (optimizer, weight_optimizer))

        def loss_and_rss():
            nonlocal forward_calls
            forward_calls += 1
            parameters = lower+p*span
            atoms = tuple(tuple(builder(parameters[block]) for block in blocks)
                          for builder, blocks in zip(builders, layouts))
            rss, penalties = [], []
            for e, (base, target, coef, weights) in enumerate(zip(bases, targets, coefficients, gamma)):
                columns = [base]
                for m, blocks in zip(candidate.mechanisms, atoms):
                    if m.groups[e] is not None:
                        columns.append(blocks[m.groups[e]][e][:, None])
                matrix = torch.cat(columns, dim=1)
                residual = target-matrix@coef
                rss.append(torch.sum(residual*residual))
                penalties.append(torch.sum(torch.abs(weights)*torch.abs(coef[list(amplitude_indices[e])])))
            loss = sum(r/y.numel() for r, y in zip(rss, targets))+budget.sparsity_multiplier*sum(penalties)
            return loss, rss

        completed, exhausted = 0, False
        for _ in range(budget.epochs_per_start):
            if (backward_calls >= budget.maximum_backward_calls
                    or time.process_time()-started_cpu >= budget.starts.maximum_cpu_seconds):
                exhausted = True
                break
            optimizer.zero_grad()
            weight_optimizer.zero_grad()
            loss, _ = loss_and_rss()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite conditional Adam objective")
            loss.backward()
            backward_calls += 1
            optimizer.step()
            weight_optimizer.step()
            with torch.no_grad():
                p.clamp_(0., 1.)
                # Repeated thresholding permits later re-entry, as in the
                # official example; nonlinear parameters are never thresholded.
                for coef, indices in zip(coefficients, amplitude_indices):
                    idx = list(indices)
                    values = coef[idx]
                    coef[idx] = torch.where(torch.abs(values) <= budget.amplitude_threshold, 0., values)
            for scheduler in schedulers:
                scheduler.step()
            completed += 1
        with torch.no_grad():
            terminal_loss, rss = loss_and_rss()
            terminal = float(terminal_loss)
            values = (lower+p*span).numpy().copy()
            c = tuple(coef.numpy().copy() for coef in coefficients)
            r = tuple(float(v) for v in rss)
        records.append({"seed": seed, "initial_fraction": fraction,
                        "completed_epochs": completed, "budget_exhausted": exhausted,
                        "terminal_training_objective": terminal,
                        "zero_mechanism_amplitudes": sum(int(np.count_nonzero(coef[list(idx)] == 0))
                                                          for coef, idx in zip(c, amplitude_indices))})
        # Select each program's own terminal objective; no truth or heldout use.
        if best is None or terminal < best[0]:
            best = (terminal, values, c, r)
        if exhausted:
            break
    if best is None:
        raise RuntimeError("No Adam restart completed before declared CPU budget")
    value, parameters, coefficients, rss = best
    status = "budget_exhausted" if (len(records) < len(budget.starts.seeds) or
                                   any(r["budget_exhausted"] for r in records)) else "epoch_budget_completed"
    # Common score is diagnostic only; it does not select an Adam restart.
    common = ScoreConvention.from_targets(view.targets)
    fit = FittedCandidate(candidate, parameters, coefficients, rss, value, common.score(rss, count),
                          count, view.scale.fingerprint, status, forward_calls,
                          time.process_time()-started_cpu, time.perf_counter()-started_wall, tuple(records))
    return {"fit": fit, "objective_identity": "sum_equation_mean_weak_residual_squared_plus_entrywise_gamma_amplitude_l1",
            "backward_calls": backward_calls, "forward_calls": forward_calls,
            "gamma_initialization": "seeded_independent_standard_normal",
            "coefficient_initialization": "training_OLS_at_declared_nonlinear_start",
            "post_fit_profile_refinement": False, "native_end_to_end_ADAM_SINDy": False}
