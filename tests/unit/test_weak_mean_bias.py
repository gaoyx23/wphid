import numpy as np
import torch
from wphid.discovery.weak_mean_bias import arrhenius_mean_bias, MeanBiasDiagnostic
from test_joint_weak_likelihood import problem


def test_bias_matches_independent_analytic_state_hessian():
    states = torch.tensor([[1100.], [.65], [.3]], dtype=torch.float64)
    parameters = torch.tensor([[7000., 1.2, 1.7]], dtype=torch.float64)
    noise = torch.tensor([.5, .001, .001], dtype=torch.float64)
    theta, m, q = parameters[0]
    def rate(point):
        return torch.exp(-theta/point[0])*point[1]**m*(1-point[2])**q
    hessian = torch.autograd.functional.hessian(rate, states[:, 0])
    expected = .5*torch.sum(torch.diag(hessian)*noise**2)
    np.testing.assert_allclose(arrhenius_mean_bias(states, parameters, noise).numpy()[0, 0], expected.numpy(), rtol=1e-13)


def test_independent_bilinear_rate_has_zero_mean_bias():
    states = torch.tensor([[1100.], [.65], [.3]], dtype=torch.float64)
    parameters = torch.tensor([[0., 1., 1.]], dtype=torch.float64)
    noise = torch.tensor([.5, .001, .001], dtype=torch.float64)
    assert arrhenius_mean_bias(states, parameters, noise).item() == 0.


def test_diagnostic_correction_sign_and_gradient_keep_parent_frozen():
    model = problem()
    wrapper = MeanBiasDiagnostic(model)
    packet = model.packets[0]
    original = packet["targets"].clone()
    kinetics, coefficients = model.unpack(model.tensor(model.initial))
    correction = wrapper.packet_correction(kinetics, coefficients, packet)
    logdet, quadratic = model.packet_terms(kinetics, coefficients, {**packet, "targets": original+correction})
    value, gradient = wrapper.value_and_gradient(model.initial)
    np.testing.assert_allclose(value, (.5*(logdet+quadratic)/model.residual_count).item(), rtol=1e-13)
    left, right = model.initial.copy(), model.initial.copy()
    left[0] -= 1e-5
    right[0] += 1e-5
    numerical = (wrapper.value_and_gradient(right)[0]-wrapper.value_and_gradient(left)[0])/2e-5
    np.testing.assert_allclose(gradient[0], numerical, rtol=2e-5)
    torch.testing.assert_close(packet["targets"], original)
