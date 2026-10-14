import numpy as np
import pytest
from wphid.weak.noise_validation import compare_noise_moments


def test_exact_centered_covariance_reference():
    x=np.concatenate([np.eye(3),-np.eye(3)])*np.sqrt(5/2)
    output=compare_noise_moments(residual_samples=x+2,baseline=np.full(3,2.),predicted_covariance=np.eye(3))
    assert output["whitened_mean_change_norm"]<1.e-14
    assert output["whitened_variance_trace_ratio"]==pytest.approx(1.)
    assert output["whitened_covariance_relative_frobenius_error"]<1.e-14


def test_mean_shift_is_not_hidden_by_centered_covariance():
    x=np.concatenate([np.eye(3),-np.eye(3)])*np.sqrt(5/2)+1
    output=compare_noise_moments(residual_samples=x,baseline=np.zeros(3),predicted_covariance=np.eye(3))
    assert output["whitened_mean_change_norm"]==pytest.approx(np.sqrt(3))
    assert output["whitened_variance_trace_ratio"]==pytest.approx(1.)


@pytest.mark.parametrize("covariance",[np.zeros((3,3)),np.full((2,2),np.nan)])
def test_bad_covariance_is_not_regularized_silently(covariance):
    with pytest.raises(ValueError):compare_noise_moments(residual_samples=np.ones((6,3)),baseline=np.zeros(3),predicted_covariance=covariance)
