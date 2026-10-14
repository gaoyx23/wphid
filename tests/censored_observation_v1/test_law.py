"""Exact censoring moments and first-order centering; no Monte Carlo samples."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from wphid.coupled_balance_v1.assembler import WeakSystem
from wphid.coupled_balance_v1.noise import SharedObservationLinearization
from wphid.coupled_likelihood_v1 import evaluate_quasi_likelihood
from wphid.censored_observation_v1 import CensoredGaussianLaw, RawMoments, center_weak_balance
from wphid.observations import _clip_channel


def law(channel="pO2", sigma=2.):
    return CensoredGaussianLaw.archived_channel(channel, sigma, origin="declared_control_noise",
        metadata_identity="artificial-qualification-only", latent_errors_independent=True)


def test_zero_pressure_has_positive_bias_and_half_gaussian_variance():
    moments = law().moments([0.])
    np.testing.assert_allclose(moments.mean_error, [2/np.sqrt(2*np.pi)])
    np.testing.assert_allclose(moments.variance, [4*(.5-1/(2*np.pi))])
    np.testing.assert_allclose(moments.lower_atom_probability, [.5])
    np.testing.assert_allclose(law().negative_log_likelihood([0.], [0.]), [np.log(2)])


def test_carbon_at_upper_limit_has_negative_bias():
    moments = law("carbon", .03).moments([1.])
    np.testing.assert_allclose(moments.mean_error, [-.03/np.sqrt(2*np.pi)])
    np.testing.assert_allclose(moments.variance, [.03**2*(.5-1/(2*np.pi))])


def test_unclipped_large_mean_small_variance_uses_error_moments_without_cancellation():
    model = CensoredGaussianLaw(.01, -np.inf, np.inf, "declared_control_noise", "unbounded-toy", True)
    moments = model.moments([1e6])
    np.testing.assert_array_equal(moments.mean_error, [0.])
    np.testing.assert_allclose(moments.variance, [.0001], rtol=1e-14)


def test_two_sided_moments_agree_with_independent_direct_quadrature():
    model = CensoredGaussianLaw(.4, 0., 1., "declared_control_noise", "bounded-toy", True)
    moments = model.moments([.3])
    first = quad(lambda x: np.clip(x, 0., 1.)*norm.pdf(x, .3, .4), -np.inf, np.inf, epsabs=1e-11)[0]
    second = quad(lambda x: (np.clip(x, 0., 1.)-.3)**2*norm.pdf(x, .3, .4), -np.inf, np.inf, epsabs=1e-11)[0]
    np.testing.assert_allclose(moments.observed_mean, [first], atol=2e-10)
    np.testing.assert_allclose(moments.variance, [second-(first-.3)**2], atol=2e-10)


def test_zero_sigma_is_explicitly_deterministic_not_a_variance_floor():
    model = law(sigma=0.)
    np.testing.assert_array_equal(model.moments([0., 3.]).variance, [0., 0.])
    np.testing.assert_array_equal(model.negative_log_likelihood([0., 3.], [0., 3.]), [0., 0.])
    assert np.isinf(model.negative_log_likelihood([1.], [0.])[0])


@pytest.mark.parametrize("channel", ["T", "pO2", "pH2O", "pProducts", "stress", "carbon", "crack"])
def test_declared_channel_bounds_match_frozen_observation_syntax(channel):
    model = law(channel, 0.)
    points = np.array([-5., 0., .01, .5, 1., 2., 300.])
    np.testing.assert_array_equal(model.moments(points).observed_mean, _clip_channel(channel, points))


def test_missing_noise_identity_or_shared_latent_errors_are_rejected():
    with pytest.raises(ValueError, match="identity"):
        CensoredGaussianLaw(1., 0., np.inf, "declared_control_noise", "", True)
    with pytest.raises(ValueError, match="shared"):
        CensoredGaussianLaw(1., 0., np.inf, "declared_control_noise", "correlated-toy", False)


def test_boundaries_use_atom_probabilities_and_interior_uses_density():
    model = CensoredGaussianLaw(.4, 0., 1., "declared_control_noise", "bounded-toy", True)
    values = model.negative_log_likelihood([0., .5, 1.], [.3, .3, .3])
    expected = [-np.log(norm.cdf(-.3/.4)), -norm.logpdf(.5, .3, .4), -np.log(norm.sf((1-.3)/.4))]
    np.testing.assert_allclose(values, expected, atol=1e-14)
    with pytest.raises(ValueError, match="outside"):
        model.negative_log_likelihood([1.1], [.3])


def test_first_order_centering_retains_target_design_cross_terms_and_conditional_gradient():
    balance = WeakSystem(np.array([.3, .4]), np.array([[.2], [.1]]), {"uncentered_outer": np.ones((2, 1))},
                         (("f", 0, 0), ("f", 0, 1)), ("a",))
    derivative = SharedObservationLinearization([[1., .2], [.1, 1.]], ([[.1, .2], [.2, .1]],), ("x0", "x1"))
    moments = RawMoments(("x0", "x1"), np.array([.2, -.1]), np.array([.4, .5]), "supplied-latent-toy", True)
    centered, factor = center_weak_balance(balance, derivative, moments)
    assert centered.matrix_jacobian == {}
    for a in (.2, .7):
        np.testing.assert_allclose(centered.residual([a]), balance.residual([a])-derivative.residual_jacobian([a]) @ moments.mean_error)
    value = evaluate_quasi_likelihood(centered, derivative, [.3], factor, compute_linear_gradient=True)
    numerical = (evaluate_quasi_likelihood(centered, derivative, [.30001], factor).objective-
                 evaluate_quasi_likelihood(centered, derivative, [.29999], factor).objective)/.00002
    np.testing.assert_allclose(value.linear_gradient, [numerical], rtol=2e-8, atol=2e-10)


def test_moment_coordinate_identity_cannot_be_reordered_silently():
    balance = WeakSystem(np.ones(2), np.ones((2, 1)), {}, (("f", 0, 0), ("f", 0, 1)), ("a",))
    derivative = SharedObservationLinearization(np.eye(2), (np.eye(2),), ("x0", "x1"))
    moments = RawMoments(("x1", "x0"), np.zeros(2), np.ones(2), "toy-law", True)
    with pytest.raises(ValueError, match="identities"):
        center_weak_balance(balance, derivative, moments)
