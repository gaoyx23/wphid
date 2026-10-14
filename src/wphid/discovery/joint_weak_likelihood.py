"""Opt-in parameter-dependent Gaussian weak-residual criterion for T/O/D.

This is a conditional PDE development adapter, not the official WENDy solver.
Known structure and calibrated independent nodal noise are required. Both
linear coefficients and kinetic parameters enter the covariance and are fit
jointly. No forward solve or true kinetic parameter enters this module.
"""
from __future__ import annotations
import numpy as np
import torch

from wphid.weak.residual_covariance import linearized_residual_covariance


class ArrheniusJointWeakLikelihood:
    def __init__(self, packets, groups, bounds, initial_parameters,
                 initial_coefficients, *, device="cpu", relative_jitter=1e-10):
        self.device = device
        self.dtype = torch.float64
        self.groups = tuple(groups)
        if len(groups) != 3 or set(groups) != set(range(max(groups)+1)):
            raise ValueError("three active canonical equation groups are required")
        initial_parameters = np.asarray(initial_parameters, dtype=float)
        initial_coefficients = np.asarray(initial_coefficients, dtype=float)
        bounds = np.asarray(bounds, dtype=float)
        if (bounds.shape != (3, 2) or initial_parameters.shape != (max(groups)+1, 3)
                or initial_coefficients.shape != (3, 4) or not packets
                or np.any(bounds[:, 1] <= bounds[:, 0])
                or not np.all(np.isfinite(initial_parameters))
                or not np.all(np.isfinite(initial_coefficients))):
            raise ValueError("finite Arrhenius parameter groups and three base/amplitude rows required")
        if not 0 < relative_jitter < 1e-3:
            raise ValueError("small positive numerical jitter required")
        self.relative_jitter = relative_jitter
        self.lower = self.tensor(bounds[:, 0])
        self.width = self.tensor(bounds[:, 1]-bounds[:, 0])
        self.group_index = torch.tensor(groups, device=device, dtype=torch.long)
        self.kinetic_count = initial_parameters.size
        # Coordinate scaling only; no coefficient sign constraints or true scales.
        coefficient_scale = np.maximum(np.abs(initial_coefficients), 1e-12)
        self.coefficient_scale = self.tensor(coefficient_scale)
        kinetic_start = (initial_parameters-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])
        self.initial = np.concatenate((kinetic_start.ravel(), (initial_coefficients/coefficient_scale).ravel()))
        self.optimization_bounds = [(0.0, 1.0)]*self.kinetic_count + [(None, None)]*12
        self.packets = []
        for packet in packets:
            states = np.asarray(packet["states"], dtype=float).reshape(3, -1)
            noise = np.asarray(packet["noise_stds"], dtype=float)
            if (noise.shape != (3,) or np.any(noise <= 0) or not np.all(np.isfinite(noise))
                    or not np.all(np.isfinite(states)) or np.any(states[:2] <= 0)
                    or np.any(states[2] >= 1)):
                raise ValueError("positive T/O, D<1, and calibrated positive state-noise scales required")
            operators = packet["operators"]
            # Fixed row conditioning changes NLL only by an additive constant
            # absent jitter. It is computed at the saved estimate, never truth.
            scales = []
            for equation, group in enumerate(groups):
                theta, m, q = initial_parameters[group]
                temperature, oxygen, damage = states
                rate = np.exp(-theta/temperature)*oxygen**m*(1-damage)**q
                gradients = (rate*theta/temperature**2, rate*m/oxygen, -rate*q/(1-damage))
                covariance = linearized_residual_covariance(operators,
                    initial_coefficients[equation, :3], initial_coefficients[equation, 3],
                    gradients, noise, equation)
                diagonal = np.diag(covariance)
                if np.any(diagonal <= 0):
                    raise ValueError("reference residual covariance diagonal must be positive")
                scales.append(1/np.sqrt(diagonal))
            converted = {"states": self.tensor(states), "noise": self.tensor(noise),
                         "bases": self.tensor(packet["bases"]), "targets": self.tensor(packet["targets"]),
                         "row_scale": self.tensor(np.asarray(scales))}
            converted.update({name: self.tensor(operators[name].toarray())
                              for name in ("mass", "time", "axial", "radial")})
            self.packets.append(converted)
        self.residual_count = sum(p["targets"].numel() for p in self.packets)

    def tensor(self, array):
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    def unpack(self, vector):
        kinetics = self.lower + vector[:self.kinetic_count].reshape(-1, 3)*self.width
        coefficients = vector[self.kinetic_count:].reshape(3, 4)*self.coefficient_scale
        return kinetics, coefficients

    def packet_terms(self, kinetics, coefficients, packet):
        temperature, oxygen, damage = packet["states"]
        theta, m, q = kinetics[self.group_index].unbind(dim=1)
        rate = torch.exp(-theta[:, None]/temperature) * oxygen**m[:, None] * (1-damage)**q[:, None]
        gradients = torch.stack((rate*theta[:, None]/temperature**2,
                                rate*m[:, None]/oxygen, -rate*q[:, None]/(1-damage)), dim=1)
        mass = packet["mass"]
        amplitudes = coefficients[:, 3]
        actions = rate @ mass.T * amplitudes[:, None]
        residual = packet["targets"] - torch.einsum("ewk,ek->ew", packet["bases"], coefficients[:, :3]) - actions
        noise_jacobians = []
        for field in range(3):
            by_equation = []
            for equation in range(3):
                derivative = -amplitudes[equation]*mass*gradients[equation, field][None, :]
                if field == equation:
                    derivative = (derivative - packet["time"] - coefficients[equation, 0]*packet["axial"]
                        - coefficients[equation, 1]*packet["radial"] - coefficients[equation, 2]*mass)
                by_equation.append(derivative*packet["row_scale"][equation, :, None]*packet["noise"][field])
            noise_jacobians.append(torch.cat(by_equation, dim=0))
        jacobian = torch.stack(noise_jacobians)
        covariance = torch.bmm(jacobian, jacobian.transpose(1, 2)).sum(dim=0)
        covariance = (covariance+covariance.T)/2
        covariance = covariance + self.relative_jitter*torch.mean(torch.diag(covariance))*torch.eye(
            covariance.shape[0], dtype=self.dtype, device=self.device)
        factor = torch.linalg.cholesky(covariance)
        scaled = (residual*packet["row_scale"]).reshape(-1, 1)
        whitened = torch.linalg.solve_triangular(factor, scaled, upper=False)
        quadratic = torch.sum(whitened**2)
        logdet = 2*torch.log(torch.diag(factor)).sum()
        return logdet, quadratic

    def loss(self, vector):
        kinetics, coefficients = self.unpack(vector)
        terms = [self.packet_terms(kinetics, coefficients, p) for p in self.packets]
        return sum(0.5*(a+b) for a, b in terms)/self.residual_count

    def value_and_gradient(self, vector):
        values = self.tensor(vector).clone().requires_grad_(True)
        loss = self.loss(values)
        gradient, = torch.autograd.grad(loss, values)
        value, derivative = float(loss.detach().cpu()), gradient.detach().cpu().numpy()
        if not np.isfinite(value) or not np.all(np.isfinite(derivative)):
            raise FloatingPointError("nonfinite weak likelihood/gradient")
        return value, derivative

    def physical_values(self, vector):
        kinetics, coefficients = self.unpack(self.tensor(vector))
        return kinetics.cpu().numpy(), coefficients.cpu().numpy()

    def decomposition(self, vector):
        with torch.no_grad():
            kinetics, coefficients = self.unpack(self.tensor(vector))
            terms = [self.packet_terms(kinetics, coefficients, p) for p in self.packets]
        return {"logdet_sum": sum(float(a.cpu()) for a, _ in terms),
                "quadratic_sum": sum(float(b.cpu()) for _, b in terms),
                "residual_count": self.residual_count,
                "normalized_half_nll": sum(0.5*float((a+b).cpu()) for a, b in terms)/self.residual_count,
                "cross_equation_covariance_used": True,
                "parameter_dependent_covariance": True}
