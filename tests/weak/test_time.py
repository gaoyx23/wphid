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
