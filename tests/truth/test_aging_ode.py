import numpy as np
import pytest

from wphid.truth.aging_ode import simulate_condition, simulate_named_condition


def test_conditions_independently_excitate_sigma_and_fatigue():
    """Removing cyclic loading or its fatigue response breaks this test."""
    low = simulate_named_condition("thermal_hold")
    cyc = simulate_named_condition("fatigue_rich")

    assert low.states["sigma"].max() > cyc.states["sigma"].mean()
    assert cyc.states["Df"][-1] > 2 * low.states["Df"][-1]


def test_named_conditions_export_only_observations_with_bounded_damage():
    """Leaking truth metadata or allowing runaway damage breaks this test."""
    for condition_id in ("thermal_hold", "fatigue_rich", "oxygen_rich"):
        trajectory = simulate_named_condition(condition_id)

        assert trajectory.condition_id == condition_id
        assert set(trajectory.states) == {"T", "pO2", "sigma", "Df", "Do"}
        np.testing.assert_array_less(-1e-12, trajectory.states["Df"] + 1e-12)
        np.testing.assert_array_less(-1e-12, trajectory.states["Do"] + 1e-12)
        assert trajectory.states["Df"].max() < 0.8
        assert trajectory.states["Do"].max() < 0.8


def test_simulate_condition_requires_aligned_observed_forcings():
    """Silently integrating misaligned forcings breaks this test."""
    t = np.linspace(0.0, 1.0, 11)
    forcing = {
        "T": np.full(t.size, 325.0),
        "pO2": np.full(t.size, 0.2),
        "sigma": np.zeros(t.size - 1),
    }

    with pytest.raises(ValueError, match="match the time axis"):
        simulate_condition("custom", t, forcing)
