import numpy as np
import pytest

from wphid.weak.empirical_noise import noise_metric, split_noise_diagnostic, whiten_noise


def test_units_do_not_change_quadratic_score():
    cov = np.array([[4., 1.], [1., 9.]])
    center = np.array([1., 2.])
    samples = np.array([[3., 4.], [5., 7.]])
    units = np.array([1.e-8, 1.e5])
    a = whiten_noise(samples, noise_metric(center=center, covariance=cov))
    b = whiten_noise(samples*units, noise_metric(center=center*units,
        covariance=cov*units[:, None]*units[None, :]))
    np.testing.assert_allclose(np.sum(a*a, axis=1), np.sum(b*b, axis=1))


def test_fit_and_threshold_cannot_read_evaluation():
    samples = np.random.default_rng(7).normal(size=(128, 3))
    original = split_noise_diagnostic(changes=samples, predicted_covariance=np.eye(3))
    samples[96:] += 100.
    altered = split_noise_diagnostic(changes=samples, predicted_covariance=np.eye(3))
    for name in original["methods"]:
        first, second = original["methods"][name], altered["methods"][name]
        np.testing.assert_array_equal(first["metric"]["center"], second["metric"]["center"])
        np.testing.assert_array_equal(first["metric"]["transform"], second["metric"]["transform"])
        assert first["calibrated_squared_radius"] == second["calibrated_squared_radius"]
        assert second["evaluation_inside_count"] == 0
    assert original["calibration_order_statistic_rank"] == 32
    assert not original["candidate_selection"]


def test_empirical_bias_translation_does_not_change_empirical_scores():
    samples = np.random.default_rng(8).normal(size=(128, 3))
    a = split_noise_diagnostic(changes=samples, predicted_covariance=np.eye(3))
    b = split_noise_diagnostic(changes=samples + [4., -3., 1.], predicted_covariance=np.eye(3))
    np.testing.assert_allclose(a["methods"]["finite_amplitude_empirical"]["evaluation_scores"],
                               b["methods"]["finite_amplitude_empirical"]["evaluation_scores"])


def test_rank_deficiency_is_explicit_not_silent():
    metric = noise_metric(center=[0., 0.], covariance=[[1., 1.], [1., 1.]])
    assert metric["floored_eigenvalue_count"] == 1


@pytest.mark.parametrize("cov", [[[0., 0.], [0., 1.]], [[1., 2.], [2., 1.]], [[1., .2], [.3, 1.]]])
def test_invalid_covariances_rejected(cov):
    with pytest.raises(ValueError):
        noise_metric(center=[0., 0.], covariance=cov)


def test_too_small_calibration_uses_infinite_not_interpolated_quantile():
    samples = np.arange(15., dtype=float).reshape(15, 1)
    result = split_noise_diagnostic(changes=samples, predicted_covariance=[[1.]],
        fit_count=8, calibration_count=2, nominal_coverage=.95)
    assert result["methods"]["first_order_zero_mean"]["infinite_radius"]


def test_overlapping_or_empty_blocks_rejected():
    with pytest.raises(ValueError):
        split_noise_diagnostic(changes=np.ones((96, 2)), predicted_covariance=np.eye(2))
