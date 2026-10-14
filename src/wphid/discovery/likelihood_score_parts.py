"""Read-only score/gradient decomposition; never an alternative estimator."""
import torch


def score_parts(model, vector):
    values = model.tensor(vector).clone().requires_grad_(True)
    kinetics, coefficients = model.unpack(values)
    terms = [model.packet_terms(kinetics, coefficients, packet) for packet in model.packets]
    logdet = sum(a for a, _ in terms) / (2 * model.residual_count)
    quadratic = sum(b for _, b in terms) / (2 * model.residual_count)
    gl, = torch.autograd.grad(logdet, values, retain_graph=True)
    gq, = torch.autograd.grad(quadratic, values)
    return {"normalized_half_logdet": float(logdet.detach().cpu()),
            "normalized_half_quadratic": float(quadratic.detach().cpu()),
            "logdet_gradient": gl.detach().cpu().numpy().tolist(),
            "quadratic_gradient": gq.detach().cpu().numpy().tolist()}
