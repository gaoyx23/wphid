import numpy as np
import pytest
from wphid.discovery.curvature_coordinates import finite_difference_hessian, curvature_summary, whitening_coordinates


def test_quadratic_hessian_and_rotation_whitening():
    matrix = np.array([[1e5, 99.], [99., 1.]])
    def criterion(point):
        return .5*point@matrix@point, matrix@point
    estimated, asymmetry = finite_difference_hessian(criterion, [1., 2.])
    np.testing.assert_allclose(estimated, matrix, rtol=1e-6)
    assert asymmetry < 1e-8
    rotation = whitening_coordinates(estimated)
    np.testing.assert_allclose(rotation.T@matrix@rotation, np.eye(2), atol=1e-6)
    assert curvature_summary(matrix)["positive_definite"]


def test_indefinite_curvature_is_reported_not_mislabeled_identifiable():
    matrix = np.diag([-1., 10.])
    result = curvature_summary(matrix)
    assert not result["positive_definite"] and result["spectral_condition_if_positive"] is None
    rotation = whitening_coordinates(matrix)
    assert np.all(np.isfinite(rotation))
    with pytest.raises(ValueError):
        whitening_coordinates(np.zeros((2, 2)))
