import numpy as np
import pytest

from wphid.discovery.profiled import fixed_parameter_profile, profiled_nonlinear_fit


def test_profiled_fit_recovers_arrhenius_parameter():
    temperature = np.linspace(500.0, 2200.0, 300)
    theta = 7200.0
    target = 1.7 + 3.2 * np.exp(-theta / temperature)
    fit = profiled_nonlinear_fit(
        np.ones((temperature.size, 1)), target,
        lambda parameter: np.exp(-parameter[0] / temperature),
        [(2000.0, 14000.0)], seed=12,
    )
    assert fit.parameters[0] == pytest.approx(theta, rel=2e-3)
    assert fit.coefficients[-1] == pytest.approx(3.2, rel=2e-3)


def test_fixed_profile_minimum_is_at_truth():
    temperature = np.linspace(600.0, 2000.0, 200)
    grid = np.linspace(4000.0, 10000.0, 121)
    target = 2.0 * np.exp(-7000.0 / temperature)
    rss, amplitude = fixed_parameter_profile(
        np.empty((temperature.size, 0)), target,
        lambda parameter: np.exp(-parameter[0] / temperature), grid,
    )
    assert grid[np.argmin(rss)] == pytest.approx(7000.0)
    assert amplitude[np.argmin(rss)] == pytest.approx(2.0)


def test_profiled_fit_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        profiled_nonlinear_fit(np.ones((3, 1)), np.ones(3),
                               lambda parameter: np.ones(3), [(1.0, 1.0)])


def test_profiled_local_search_never_evaluates_outside_bounds():
    x = np.linspace(0.0, 1.0, 30)
    visited = []

    def atom(parameters):
        visited.append(float(parameters[0]))
        return np.exp(-parameters[0] * x)

    profiled_nonlinear_fit(
        np.ones((x.size, 1)), np.exp(-0.7 * x), atom,
        [(0.0, 1.0)], seed=3, global_iterations=2,
    )
    assert visited
    assert min(visited) >= 0.0
    assert max(visited) <= 1.0
