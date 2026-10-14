import numpy as np
import pytest
from test_joint_weak_likelihood import problem
from wphid.discovery.likelihood_score_parts import score_parts
from wphid.discovery.weak_bootstrap import centered_measurement_bootstrap


def test_parts_sum_to_original_value_and_gradient():
    model = problem()
    value, gradient = model.value_and_gradient(model.initial)
    p = score_parts(model, model.initial)
    assert p["normalized_half_logdet"] + p["normalized_half_quadratic"] == pytest.approx(value)
    np.testing.assert_allclose(np.asarray(p["logdet_gradient"]) + p["quadratic_gradient"], gradient, atol=1e-9)


def test_zero_residual_not_zero_declared_variance_or_zero_total_gradient():
    model = problem()
    zero = centered_measurement_bootstrap(model, model.initial, 0, noise_multiplier=0.)
    p = score_parts(zero, model.initial)
    assert p["normalized_half_quadratic"] < 1e-20
    assert np.linalg.norm(p["quadratic_gradient"]) < 1e-8
    assert np.linalg.norm(p["logdet_gradient"]) > 1e-6
