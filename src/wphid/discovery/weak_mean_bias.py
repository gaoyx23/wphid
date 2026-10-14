"""Second-order mean-only diagnostic for independent Gaussian state noise.

Not a full second-order likelihood: covariance remains the parent's first-order
one. This quantifies one omitted term without changing the frozen estimator.
"""
import numpy as np
import torch


def arrhenius_mean_bias(states, equation_parameters, noise_stds):
    temperature, oxygen, damage = states
    theta, m, q = equation_parameters.unbind(dim=1)
    theta, m, q = theta[:, None], m[:, None], q[:, None]
    rate = torch.exp(-theta/temperature)*oxygen**m*(1-damage)**q
    relative = ((theta**2/temperature**4-2*theta/temperature**3)*noise_stds[0]**2
                +m*(m-1)/oxygen**2*noise_stds[1]**2
                +q*(q-1)/(1-damage)**2*noise_stds[2]**2)
    return .5*rate*relative


class MeanBiasDiagnostic:
    def __init__(self, model):
        self.model = model

    def packet_correction(self, kinetics, coefficients, packet):
        bias = arrhenius_mean_bias(packet["states"], kinetics[self.model.group_index], packet["noise"])
        return coefficients[:, 3, None]*(bias@packet["mass"].T)

    def loss(self, vector):
        model = self.model
        kinetics, coefficients = model.unpack(vector)
        terms = []
        for packet in model.packets:
            # E[r(y+noise)] ~ -A M bias; r + A M bias restores zero mean.
            correction = self.packet_correction(kinetics, coefficients, packet)
            corrected = {**packet, "targets": packet["targets"]+correction}
            terms.append(model.packet_terms(kinetics, coefficients, corrected))
        return sum(.5*(a+b) for a, b in terms)/model.residual_count

    def value_and_gradient(self, vector):
        values = self.model.tensor(vector).clone().requires_grad_(True)
        loss = self.loss(values)
        gradient, = torch.autograd.grad(loss, values)
        value, gradient = float(loss.detach().cpu()), gradient.detach().cpu().numpy()
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("nonfinite mean-bias diagnostic")
        return value, gradient
