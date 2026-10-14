import numpy as np
import pytest

from wphid.truth.aging_ode import (
    realistic_oxidation_evaluation_truth,
    simulate_condition,
    simulate_mission_profile,
    simulate_named_condition,
    simulate_realistic_condition,
)


def test_conditions_independently_excitate_sigma_and_fatigue():
    """Removing cyclic loading or its fatigue response breaks this test."""
    low = simulate_named_condition("thermal_hold")
    cyc = simulate_named_condition("fatigue_rich")

    assert low.states["sigma"].max() > cyc.states["sigma"].mean()
    assert cyc.states["Df"][-1] > 2 * low.states["Df"][-1]


def test_oxygen_rich_condition_independently_excites_oxygen():
    """Replacing oxygen_rich with a low-oxygen forcing breaks this test."""
    oxygen = simulate_named_condition("oxygen_rich")
    thermal = simulate_named_condition("thermal_hold")
    fatigue = simulate_named_condition("fatigue_rich")

    assert oxygen.states["pO2"].min() > max(
        thermal.states["pO2"].max(), fatigue.states["pO2"].max()
    )
    assert oxygen.states["Do"][-1] > 2 * fatigue.states["Do"][-1]


def test_named_condition_simulation_is_repeatably_deterministic():
    """Introducing stochastic forcing or solver state breaks this test."""
    first = simulate_named_condition("fatigue_rich")
    second = simulate_named_condition("fatigue_rich")

    assert first.condition_id == second.condition_id
    np.testing.assert_array_equal(first.t, second.t)
    for name in first.states:
        np.testing.assert_array_equal(first.states[name], second.states[name])


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


def test_named_conditions_are_complete_rocket_mission_cycles():
    """Every legacy condition ID must now span all four mission phases."""
    for condition_id in ("thermal_hold", "fatigue_rich", "oxygen_rich"):
        trajectory = simulate_named_condition(condition_id)
        temperature = trajectory.states["T"]
        baseline = temperature[0]
        peak_index = int(np.argmax(temperature))

        # Preheat precedes the rapid ignition ramp; a modulated high-temperature
        # dwell follows, then shutdown cooling returns toward baseline.
        assert temperature[peak_index // 2] > baseline + 75.0
        assert temperature[peak_index] - temperature[peak_index - 1] > 10.0
        assert np.ptp(temperature[peak_index : peak_index + 20]) > 5.0
        assert temperature[-1] < temperature[peak_index] - 500.0

        # Representative material-wall temperatures are neither cryogenic nor
        # chamber-gas temperatures.
        assert 300.0 <= temperature.min() <= 500.0
        assert 900.0 <= temperature.max() <= 1_500.0

_CYCLE_PROFILE = {
    "baseline_temperature": 340.0,
    "peak_temperature": 1100.0,
    "preheat_duration": 2.4,
    "ignition_duration": 0.6,
    "dwell_duration": 3.7,
    "oxygen_pressure": 0.18,
    "load": 30.0,
    "load_amplitude": 8.0,
}


def test_initial_damage_creates_independent_history_with_identical_forcing():
    fresh = simulate_mission_profile("fresh", _CYCLE_PROFILE)
    cycled = simulate_mission_profile(
        "cycled", _CYCLE_PROFILE, initial_damage=(0.12, 0.0)
    )

    for name in ("T", "pO2", "sigma"):
        np.testing.assert_array_equal(fresh.states[name], cycled.states[name])
    np.testing.assert_allclose(
        (fresh.states["Df"][0], fresh.states["Do"][0]), (0.0, 0.0)
    )
    np.testing.assert_allclose(
        (cycled.states["Df"][0], cycled.states["Do"][0]), (0.12, 0.0)
    )
    assert cycled.states["Df"][-1] > fresh.states["Df"][-1]
    assert cycled.states["Do"][-1] > fresh.states["Do"][-1]


@pytest.mark.parametrize(
    "initial_damage",
    (
        (0.0,),
        (0.0, 0.0, 0.0),
        (-0.01, 0.0),
        (0.0, 0.8),
        (np.nan, 0.0),
        (0.0, np.inf),
    ),
)
def test_simulate_condition_rejects_invalid_initial_damage(initial_damage):
    time = np.linspace(0.0, 1.0, 11)
    forcing = {
        "T": np.full(time.size, 900.0),
        "pO2": np.full(time.size, 0.1),
        "sigma": np.zeros(time.size),
    }

    with pytest.raises(ValueError, match="initial_damage"):
        simulate_condition(
            "invalid-history", time, forcing, initial_damage=initial_damage
        )


def test_named_conditions_keep_zero_damage_default():
    trajectory = simulate_named_condition("thermal_hold")

    assert trajectory.states["Df"][0] == 0.0
    assert trajectory.states["Do"][0] == 0.0


_REALISTIC_HOT_PROFILE = {
    "baseline_temperature": 340.0,
    "peak_temperature": 1430.0,
    "preheat_duration": 2.4,
    "ignition_duration": 0.6,
    "dwell_duration": 4.6,
    "oxygen_pressure": 0.15,
    "load": 80.0,
    "load_amplitude": 40.0,
}
_REALISTIC_COOL_PROFILE = {
    "baseline_temperature": 330.0,
    "peak_temperature": 1160.0,
    "preheat_duration": 2.7,
    "ignition_duration": 0.8,
    "dwell_duration": 4.0,
    "oxygen_pressure": 0.45,
    "load": 200.0,
    "load_amplitude": 50.0,
}


def test_realistic_condition_resolves_hot_dwell_oxidation():
    """The adaptive solver must resolve the hot dwell. Without ``max_step`` the
    cold-phase rates (~1e-32) let LSODA jump across the active window in one
    step, resolving ~zero oxidation damage."""
    trajectory = simulate_realistic_condition("hot", _REALISTIC_HOT_PROFILE)
    assert trajectory.states["T"].max() > 1400.0
    assert trajectory.states["Do"][-1] - trajectory.states["Do"][0] > 0.01
    assert trajectory.states["Df"][-1] > trajectory.states["Df"][0]


def test_realistic_condition_is_deterministic():
    first = simulate_realistic_condition("hot", _REALISTIC_HOT_PROFILE)
    second = simulate_realistic_condition("hot", _REALISTIC_HOT_PROFILE)
    for name in first.states:
        np.testing.assert_array_equal(first.states[name], second.states[name])


def test_realistic_interaction_scale_default_is_exactly_backward_compatible():
    implicit = simulate_realistic_condition("implicit", _REALISTIC_HOT_PROFILE)
    explicit = simulate_realistic_condition(
        "explicit", _REALISTIC_HOT_PROFILE, interaction_scale=1.0
    )
    for name in implicit.states:
        np.testing.assert_array_equal(implicit.states[name], explicit.states[name])


def test_realistic_interaction_scale_zero_is_blind_negative_control():
    positive = simulate_realistic_condition(
        "positive", _REALISTIC_HOT_PROFILE, initial_damage=(0.55, 0.0)
    )
    negative = simulate_realistic_condition(
        "negative",
        _REALISTIC_HOT_PROFILE,
        initial_damage=(0.55, 0.0),
        interaction_scale=0.0,
    )
    for name in ("T", "pO2", "sigma"):
        np.testing.assert_array_equal(positive.states[name], negative.states[name])
    np.testing.assert_allclose(positive.states["Df"], negative.states["Df"], rtol=1e-7)
    assert negative.states["Do"][-1] < positive.states["Do"][-1]


@pytest.mark.parametrize("value", (-1.0, np.nan, np.inf, True, "bad"))
def test_realistic_condition_rejects_invalid_interaction_scale(value):
    with pytest.raises(ValueError, match="interaction_scale"):
        simulate_realistic_condition(
            "bad-scale", _REALISTIC_HOT_PROFILE, interaction_scale=value
        )


def test_realistic_condition_exports_bounded_observations():
    for profile in (_REALISTIC_HOT_PROFILE, _REALISTIC_COOL_PROFILE):
        trajectory = simulate_realistic_condition("cond", profile)
        assert set(trajectory.states) == {"T", "pO2", "sigma", "Df", "Do"}
        np.testing.assert_array_less(-1e-12, trajectory.states["Df"] + 1e-12)
        np.testing.assert_array_less(-1e-12, trajectory.states["Do"] + 1e-12)
        assert trajectory.states["Df"].max() < 0.8
        assert trajectory.states["Do"].max() < 0.8


def test_realistic_condition_spans_active_temperature_band():
    hot = simulate_realistic_condition("hot", _REALISTIC_HOT_PROFILE)
    cool = simulate_realistic_condition("cool", _REALISTIC_COOL_PROFILE)
    assert hot.states["T"].max() > cool.states["T"].max()
    assert cool.states["T"].max() >= 1100.0
    assert hot.states["Do"][-1] > cool.states["Do"][-1]


def test_realistic_condition_respects_initial_damage():
    fresh = simulate_realistic_condition("fresh", _REALISTIC_HOT_PROFILE)
    cycled = simulate_realistic_condition(
        "cycled", _REALISTIC_HOT_PROFILE, initial_damage=(0.3, 0.0)
    )
    assert fresh.states["Df"][0] == 0.0
    assert cycled.states["Df"][0] == 0.3
    assert cycled.states["Do"][-1] > fresh.states["Do"][-1]


def test_realistic_oxidation_evaluation_truth_exposes_calibrated_constants():
    truth = realistic_oxidation_evaluation_truth()
    assert truth["arrhenius_theta"] == 25000.0
    assert truth["coefficient_by_factor"]["pO2"] == 5.0e5
    assert truth["coefficient_by_factor"]["pO2*sigma"] == 3.0e3
    assert truth["coefficient_by_factor"]["pO2*Df"] == 8.0e5


def test_realistic_condition_rejects_invalid_profile():
    bad = dict(_REALISTIC_HOT_PROFILE)
    bad["peak_temperature"] = 200.0
    with pytest.raises(ValueError, match="peak_temperature"):
        simulate_realistic_condition("bad", bad)
