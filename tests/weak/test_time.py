import numpy as np
import pytest

from wphid.weak.time import project_time_weak


def test_weak_projection_recovers_linear_decay_without_derivatives():
    """Using pointwise derivatives or the wrong integration sign breaks recovery."""
    t = np.linspace(0, 4, 401)
    y = np.exp(-2 * t)

    A, b = project_time_weak(t, y, y[:, None], window=81, stride=20)
    coef = np.linalg.lstsq(A, b, rcond=None)[0]

    assert coef[0] == pytest.approx(-2.0, rel=2e-3)


def test_weak_projection_recovers_linear_decay_with_five_percent_noise():
    """Removing weak averaging or using noisy pointwise derivatives breaks recovery."""
    t = np.linspace(0, 4, 401)
    clean = np.exp(-2 * t)
    relative_errors = []

    for seed in range(20):
        rng = np.random.default_rng(seed)
        observed = clean * (1.0 + 0.05 * rng.standard_normal(clean.size))
        A, b = project_time_weak(t, observed, observed[:, None], window=81, stride=20)
        estimate = np.linalg.lstsq(A, b, rcond=None)[0][0]
        relative_errors.append(abs(estimate + 2.0) / 2.0)

    assert np.median(relative_errors) < 0.08


def test_weak_projection_recovers_two_feature_polynomial_model():
    """Collapsing the feature axis or mixing feature columns breaks this test."""
    t = np.linspace(0.0, 3.0, 301)
    y = t**2
    features = np.column_stack((np.ones_like(t), t))

    A, b = project_time_weak(t, y, features, window=61, stride=30)
    coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

    assert A.shape[1] == 2
    np.testing.assert_allclose(coefficients, [0.0, 2.0], atol=2e-5)


def test_weak_projection_supports_nonuniform_time_grids():
    """Replacing time-aware quadrature with fixed sample spacing breaks this test."""
    coordinate = np.linspace(0.0, 1.0, 401)
    t = 4.0 * coordinate**1.4
    y = np.exp(-2.0 * t)

    A, b = project_time_weak(t, y, y[:, None], window=81, stride=20)
    coefficient = np.linalg.lstsq(A, b, rcond=None)[0][0]

    assert coefficient == pytest.approx(-2.0, rel=3e-3)


def test_weak_projection_uses_only_complete_windows():
    """Including a trailing partial window breaks the output-row contract."""
    t = np.linspace(0.0, 1.0, 12)
    y = t**2
    features = (2.0 * t)[:, None]

    A, b = project_time_weak(t, y, features, window=5, stride=4)

    assert A.shape == (2, 1)
    assert b.shape == (2,)


@pytest.mark.parametrize(
    ("t", "y", "features", "window", "stride", "message"),
    [
        (np.array([0.0, 1.0, 0.5]), np.ones(3), np.ones((3, 1)), 3, 1, "strictly increasing"),
        (np.arange(4.0), np.ones((4, 1)), np.ones((4, 1)), 3, 1, "one-dimensional and match"),
        (np.arange(4.0), np.ones(4), np.ones(4), 3, 1, "shape"),
        (np.arange(4.0), np.ones(4), np.ones((4, 1)), 2, 1, "window"),
        (np.arange(4.0), np.ones(4), np.ones((4, 1)), 3, 0, "stride"),
    ],
)
def test_weak_projection_rejects_invalid_inputs(t, y, features, window, stride, message):
    """Accepting malformed arrays or invalid window controls breaks this test."""
    with pytest.raises(ValueError, match=message):
        project_time_weak(t, y, features, window=window, stride=stride)
