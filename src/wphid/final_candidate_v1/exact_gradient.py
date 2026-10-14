"""Exact CPU Jacobian for fixed-observation weak atoms.

This changes computational differentiation only. It neither changes the
model/loss nor supplies truth parameters, parameter profiles or new information.
The original full PyTorch path remains available and unchanged for comparison.
"""

import numpy as np
from scipy.sparse import block_diag

from .problem import WeakView


class ExactGradientView(WeakView):
    def torch_builder(self, family):
        import torch
        states = tuple(np.concatenate([c.states[i] for c in self.conditions]) for i in range(3))
        t, o, d = states
        log_o, log_d, log_remaining = np.log(o), np.log(d), np.log1p(-d)
        name = family.name
        prefactor = np.ones_like(t)
        if name in {"damage_power", "oxygen_damage_power"}:
            features = log_d[:, None]
            if name == "oxygen_damage_power":
                prefactor = o
        elif name in {"damage_logistic_power", "remaining_life_power", "oxygen_remaining_power"}:
            features = log_remaining[:, None]
            if name == "damage_logistic_power":
                prefactor = d
            elif name == "oxygen_remaining_power":
                prefactor = o
        elif name == "oxygen_power_damage":
            features, prefactor = log_o[:, None], d
        elif name == "arrhenius_oxygen_damage":
            features = np.column_stack((-1/t, log_o, log_remaining))
        elif name == "arrhenius_oxygen":
            features = np.column_stack((-1/t, log_o))
        elif name == "arrhenius_damage":
            features = np.column_stack((-1/t, log_remaining))
        else:
            raise ValueError("No exact weak-atom Jacobian for family")
        projection = block_diag([c.projection for c in self.conditions], format="csr")
        projection.eliminate_zeros()
        factors = tuple(w/n for w, n in zip(self.scale.target_weights, self.scale.norms_for(family)))

        class FixedWeakAtom(torch.autograd.Function):
            @staticmethod
            def forward(ctx, parameters):
                if parameters.device.type != "cpu" or parameters.dtype != torch.float64:
                    raise ValueError("Exact weak-atom path requires CPU float64")
                p = parameters.detach().numpy()
                rate = prefactor*np.exp(features@p)
                atom = np.asarray(projection@rate)
                jacobian = np.asarray(projection@(rate[:, None]*features))
                ctx.save_for_backward(torch.from_numpy(jacobian))
                return torch.from_numpy(atom)

            @staticmethod
            def backward(ctx, grad_output):
                (jacobian,) = ctx.saved_tensors
                return jacobian.T@grad_output

        def build(parameters):
            raw = FixedWeakAtom.apply(parameters)
            return tuple(raw*f for f in factors)
        return build
