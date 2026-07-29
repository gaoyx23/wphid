"""Private aging dynamics used to generate observation-only trajectories."""

from collections.abc import Mapping

import numpy as np
from scipy.integrate import solve_ivp

from wphid.types import Trajectory


_DAMAGE_LIMIT = 0.8
_FATIGUE_RATE = 5.0e-7
_OXIDATION_RATE = 0.18
_STRESS_OXIDATION_RATE = 6.0e-3
_FATIGUE_OXIDATION_RATE = 0.45
_ACTIVATION_TEMPERATURE = 1800.0


def simulate_condition(
    condition_id: str, t: np.ndarray, forcing: dict[str, np.ndarray]
) -> Trajectory:
    """Simulate one condition and return its observed trajectory.

    The forcing must contain one-dimensional ``T``, ``pO2``, and ``sigma``
    arrays aligned to the strictly increasing time axis.
    """
    time, observed_forcing = _validate_forcing(t, forcing)
    stress_reference = observed_forcing["sigma"][0]

    def rhs(time_value: float, damage: np.ndarray) -> np.ndarray:
        temperature = np.interp(time_value, time, observed_forcing["T"])
        oxygen_pressure = np.interp(time_value, time, observed_forcing["pO2"])
        stress = np.interp(time_value, time, observed_forcing["sigma"])
        stress_range = abs(stress - stress_reference)
        fatigue_damage, oxidation_damage = damage
        thermal_factor = np.exp(-_ACTIVATION_TEMPERATURE / temperature)
        fatigue_rate = _FATIGUE_RATE * stress_range**3 / (1.0 - fatigue_damage)
        oxidation_rate = thermal_factor * oxygen_pressure * (
            _OXIDATION_RATE
            + _STRESS_OXIDATION_RATE * stress
            + _FATIGUE_OXIDATION_RATE * fatigue_damage
        )
        return np.array([fatigue_rate, oxidation_rate])

    solution = solve_ivp(
        rhs,
        (time[0], time[-1]),
        np.zeros(2),
        t_eval=time,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"aging simulation failed: {solution.message}")
    if np.any(solution.y < 0.0) or np.any(solution.y >= _DAMAGE_LIMIT):
        raise ValueError("forcing produces damage outside the supported range")

    return Trajectory(
        t=time,
        states={
            **observed_forcing,
            "Df": solution.y[0],
            "Do": solution.y[1],
        },
        condition_id=condition_id,
    )


def simulate_named_condition(condition_id: str) -> Trajectory:
    """Return one deterministic benchmark condition by identifier."""
    time = np.linspace(0.0, 12.0, 241)
    phase = 2.0 * np.pi * time / time[-1]
    profiles = {
        "thermal_hold": {
            "T": 415.0 + 2.0 * np.sin(phase),
            "pO2": 0.20 + 0.01 * np.cos(phase),
            "sigma": np.full_like(time, 120.0),
        },
        "fatigue_rich": {
            "T": 340.0 + 4.0 * np.sin(phase),
            "pO2": 0.18 + 0.02 * np.cos(phase),
            "sigma": 55.0 + 35.0 * np.sin(5.0 * phase),
        },
        "oxygen_rich": {
            "T": 360.0 + 8.0 * np.sin(phase),
            "pO2": 0.70 + 0.10 * np.cos(2.0 * phase),
            "sigma": 20.0 + 5.0 * np.sin(3.0 * phase),
        },
    }
    try:
        forcing = profiles[condition_id]
    except KeyError as error:
        choices = ", ".join(profiles)
        raise ValueError(f"unknown condition {condition_id!r}; choose from {choices}") from error
    return simulate_condition(condition_id, time, forcing)


def _validate_forcing(
    t: np.ndarray, forcing: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time = np.asarray(t, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time axis must contain at least two strictly increasing values")
    if not np.all(np.isfinite(time)):
        raise ValueError("time axis must be finite")

    required = ("T", "pO2", "sigma")
    missing = set(required).difference(forcing)
    if missing:
        raise ValueError(f"forcing is missing required fields: {', '.join(sorted(missing))}")

    observed = {}
    for name in required:
        values = np.asarray(forcing[name], dtype=float)
        if values.ndim != 1 or values.shape != time.shape:
            raise ValueError(f"forcing {name!r} must match the time axis")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"forcing {name!r} must be finite")
        observed[name] = values
    if np.any(observed["T"] <= 0.0):
        raise ValueError("forcing 'T' must be positive")
    if np.any(observed["pO2"] < 0.0):
        raise ValueError("forcing 'pO2' must be non-negative")
    return time, observed
