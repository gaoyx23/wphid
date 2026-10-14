"""Plug-in measurement-noise bootstrap of conditional weak balances.

Reference states are observed fields, and original weak residuals are centered
at a saved fitted model. This statistical surrogate is NOT a new engineering
trajectory and does not certify the original physical noise distribution.
"""
import copy
import numpy as np
import torch


def centered_measurement_bootstrap(model, reference_vector, seed, *, noise_multiplier=1.):
    if noise_multiplier < 0:
        raise ValueError("nonnegative bootstrap noise multiplier required")
    rng = np.random.default_rng(seed)
    reference = copy.copy(model)
    reference.packets = []
    kinetics, coefficients = model.unpack(model.tensor(reference_vector))
    with torch.no_grad():
        for packet in model.packets:
            temperature, oxygen, damage = packet["states"]
            theta, m, q = kinetics[model.group_index].unbind(dim=1)
            rate = torch.exp(-theta[:, None]/temperature)*oxygen**m[:, None]*(1-damage)**q[:, None]
            mean_target = (torch.einsum("ewk,ek->ew", packet["bases"], coefficients[:, :3])
                           +coefficients[:, 3, None]*(rate@packet["mass"].T))
            delta = model.tensor(rng.normal(size=tuple(packet["states"].shape)))*packet["noise"][:, None]*noise_multiplier
            states = packet["states"]+delta
            if torch.any(states[:2] <= 0) or torch.any(states[2] >= 1):
                raise ValueError("bootstrap state leaves the declared physical function domain; no redraw")
            bases = packet["bases"]+torch.stack([delta@packet[k].T for k in ("axial", "radial", "mass")], dim=-1)
            targets = mean_target-delta@packet["time"].T
            reference.packets.append({**packet, "states": states, "bases": bases, "targets": targets})
    return reference
