import numpy as np
import pytest
from wphid.weak.grouped_noise import grouped_noise_diagnostic


def test_declared_correlations_are_preserved():
    samples = np.random.default_rng(42).normal(size=(128, 3))*[1., 2., 3.]
    reference = np.array([[1., .2, .1], [.2, 1., .3], [.1, .3, 1.]])
    result = grouped_noise_diagnostic(changes=samples, reference_covariance=reference,
        groups=["oxygen", "steam", "product"])
    metric = result["metric"]
    # Eigenvectors come from the reference correlation, up to sign.
    np.testing.assert_allclose(metric["correlation_eigenvalues"], np.linalg.eigvalsh(reference))
    assert result["estimated_covariance_parameter_count"] == 3
    assert result["reference_correlations_preserved"]


def test_evaluation_cannot_change_gains_or_calibration():
    samples = np.random.default_rng(43).normal(size=(128, 3))
    a = grouped_noise_diagnostic(changes=samples, reference_covariance=np.eye(3), groups=["a"]*3)
    samples[96:] += 100.
    b = grouped_noise_diagnostic(changes=samples, reference_covariance=np.eye(3), groups=["a"]*3)
    assert a["group_variance_gains"] == b["group_variance_gains"]
    assert a["calibrated_squared_radius"] == b["calibrated_squared_radius"]
    assert b["evaluation_inside_count"] == 0


def test_group_gain_is_pooled_normalized_variance():
    samples = np.arange(128., dtype=float)[:, None]*np.array([[1., 2.]])
    result = grouped_noise_diagnostic(changes=samples, reference_covariance=np.diag([1., 4.]), groups=["same"]*2)
    assert result["group_variance_gains"]["same"] == pytest.approx(np.var(np.arange(64.), ddof=1))


def test_invalid_group_labels_rejected():
    with pytest.raises(ValueError):
        grouped_noise_diagnostic(changes=np.ones((128, 3)), reference_covariance=np.eye(3), groups=["a"])
