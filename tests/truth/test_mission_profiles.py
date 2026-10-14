"""Experiment-declarable reusable-rocket mission profile coverage."""

import numpy as np
import pytest

from wphid.truth.aging_ode import (
    simulate_condition,
    simulate_mission_profile,
)
from wphid.truth.identifiability import diagnose_arrhenius_identifiability


# These profiles are intentionally plain mappings: an experiment configuration
# can declare them without importing truth-model implementation details.
_REFERENCE_PROFILE = {
    "baseline_temperature": 340.0,
    "peak_temperature": 900.0,
    "preheat_duration": 2.4,
    "ignition_duration": 0.6,
    "dwell_duration": 3.7,
    "oxygen_pressure": 0.18,
    "load": 30.0,
    "load_amplitude": 8.0,
}

EXPERIMENT_PROFILES = {
    "low_peak_reference": _REFERENCE_PROFILE,
    "high_peak": {**_REFERENCE_PROFILE, "peak_temperature": 1450.0},
    "slow_ignition": {**_REFERENCE_PROFILE, "ignition_duration": 0.95},
    "long_hot_dwell": {**_REFERENCE_PROFILE, "dwell_duration": 5.2},
    "cyclic_load": {**_REFERENCE_PROFILE, "load": 42.0, "load_amplitude": 24.0},
    "oxygen_rich": {**_REFERENCE_PROFILE, "oxygen_pressure": 0.48},
}


@pytest.mark.parametrize("condition_id, profile", EXPERIMENT_PROFILES.items())
def test_declared_mission_profiles_cover_a_complete_safe_duty_cycle(
    condition_id: str, profile: dict[str, float]
):
    """Each experiment profile covers heat-up, dwell, and shutdown cooling."""
    trajectory = simulate_mission_profile(condition_id, profile)
    temperature = trajectory.states["T"]

    assert 890.0 <= temperature.max() <= 1450.0
    assert temperature[int(np.argmax(temperature))] > temperature[0] + 400.0
    assert np.ptp(temperature[temperature > temperature.max() - 10.0]) > 0.0
    assert temperature[-1] < temperature.max() - 350.0
    assert trajectory.states["Df"].max() < 0.8
    assert trajectory.states["Do"].max() < 0.8


def test_declared_profiles_independently_excite_mission_drivers():
    """One-factor variants keep temperature, dwell, load, and oxygen separable."""
    reference = EXPERIMENT_PROFILES["low_peak_reference"]

    assert EXPERIMENT_PROFILES["high_peak"]["peak_temperature"] > reference["peak_temperature"]
    assert EXPERIMENT_PROFILES["slow_ignition"]["ignition_duration"] > reference["ignition_duration"]
    assert EXPERIMENT_PROFILES["long_hot_dwell"]["dwell_duration"] > reference["dwell_duration"]
    assert EXPERIMENT_PROFILES["cyclic_load"]["load_amplitude"] > reference["load_amplitude"]
    assert EXPERIMENT_PROFILES["oxygen_rich"]["oxygen_pressure"] > reference["oxygen_pressure"]

    trajectories = {
        name: simulate_mission_profile(name, profile)
        for name, profile in EXPERIMENT_PROFILES.items()
    }
    assert trajectories["slow_ignition"].states["T"][60] < trajectories[
        "low_peak_reference"
    ].states["T"][60]
    assert trajectories["long_hot_dwell"].states["T"][140] > trajectories[
        "low_peak_reference"
    ].states["T"][140]
    assert trajectories["cyclic_load"].states["Df"][-1] > trajectories[
        "low_peak_reference"
    ].states["Df"][-1]
    assert trajectories["oxygen_rich"].states["Do"][-1] > trajectories[
        "low_peak_reference"
    ].states["Do"][-1]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("baseline_temperature", 0.0, "baseline_temperature"),
        ("peak_temperature", 700.0, "peak_temperature"),
        ("dwell_duration", 0.0, "dwell_duration"),
        ("oxygen_pressure", -0.01, "oxygen_pressure"),
        ("load", -1.0, "load"),
    ),
)
def test_mission_profile_rejects_invalid_required_physical_inputs(
    field: str, value: float, message: str
):
    profile = {**_REFERENCE_PROFILE, field: value}

    with pytest.raises(ValueError, match=message):
        simulate_mission_profile("invalid", profile)


def test_mission_profiles_improve_arrhenius_excitation_over_a_narrow_temperature_band():
    mission_trajectories = tuple(
        simulate_mission_profile(name, profile)
        for name, profile in EXPERIMENT_PROFILES.items()
    )
    time = np.linspace(0.0, 12.0, 241)
    narrow_trajectories = tuple(
        simulate_condition(
            f"narrow-{index}",
            time,
            {
                "T": 1200.0 + 12.0 * np.sin(2.0 * np.pi * time / time[-1]),
                "pO2": np.full(time.size, 0.18),
                "sigma": np.full(time.size, 30.0),
            },
        )
        for index in range(2)
    )
    theta = (1200.0, 1800.0, 2400.0)
    mission = diagnose_arrhenius_identifiability(mission_trajectories, theta)
    narrow = diagnose_arrhenius_identifiability(narrow_trajectories, theta)

    assert mission["temperature_span"]["span"] >= 550.0
    assert mission["standardized_condition_number"] < (
        narrow["standardized_condition_number"] * 0.5
    )
    assert mission["profile_score_separation"] >= narrow["profile_score_separation"]
