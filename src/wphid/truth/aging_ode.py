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

# Realistic oxidation regime: Ea/R = 25000 K (Ea ~ 208 kJ/mol, the Ni-superalloy
# oxidation range), so the Arrhenius factor is sharply temperature-activated and
# oxidation is confined to the hot dwell. Coefficients are calibrated so the
# hottest condition gains ~0.02 Do per mission (design life ~38 missions) and the
# fatigue-heavy condition gains ~0.01 Df per mission (~76 missions).
_REALISTIC_ACTIVATION_TEMPERATURE = 25000.0
_REALISTIC_OXIDATION_C0 = 5.0e5
_REALISTIC_OXIDATION_C1 = 3.0e3
_REALISTIC_OXIDATION_C2 = 8.0e5
_REALISTIC_FATIGUE_RATE = 1.0e-8
# The adaptive solver must resolve the ~5 s hot dwell. Cold-phase rates are
# ~1e-32, so without a fixed maximum step LSODA takes one giant step across the
# active window and resolves ~zero oxidation damage.
_REALISTIC_SOLVER_MAX_STEP = 0.05


def simulate_condition(
    condition_id: str,
    t: np.ndarray,
    forcing: dict[str, np.ndarray],
    *,
    initial_damage: tuple[float, float] = (0.0, 0.0),
) -> Trajectory:
    """Simulate one condition and return its observed trajectory.

    The forcing must contain one-dimensional ``T``, ``pO2``, and ``sigma``
    arrays aligned to the strictly increasing time axis. ``initial_damage`` is
    the incoming ``(Df, Do)`` history from preceding mission cycles.
    """
    time, observed_forcing = _validate_forcing(t, forcing)
    initial_state = _validate_initial_damage(initial_damage)
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
        initial_state,
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


_MISSION_DURATION = 12.0
_MISSION_SAMPLES = 241
_MISSION_PROFILE_FIELDS = (
    "baseline_temperature",
    "peak_temperature",
    "preheat_duration",
    "ignition_duration",
    "dwell_duration",
    "oxygen_pressure",
    "load",
    "load_amplitude",
)
_MISSION_PROFILES = {
    # The historic IDs are retained so existing experiment configuration remains
    # valid while each now denotes a complete reusable-nozzle duty cycle.
    "thermal_hold": {
        "baseline_temperature": 340.0,
        "peak_temperature": 1425.0,
        "preheat_duration": 2.4,
        "ignition_duration": 0.6,
        "dwell_duration": 5.0,
        "oxygen_pressure": 0.13,
        "load": 65.0,
        "load_amplitude": 2.0,
    },
    "fatigue_rich": {
        "baseline_temperature": 335.0,
        "peak_temperature": 1280.0,
        "preheat_duration": 2.4,
        "ignition_duration": 0.6,
        "dwell_duration": 4.8,
        "oxygen_pressure": 0.09,
        "load": 28.0,
        "load_amplitude": 20.0,
    },
    "oxygen_rich": {
        "baseline_temperature": 350.0,
        "peak_temperature": 1180.0,
        "preheat_duration": 2.4,
        "ignition_duration": 0.6,
        "dwell_duration": 4.3,
        "oxygen_pressure": 0.45,
        "load": 14.0,
        "load_amplitude": 3.0,
    },
}


def simulate_mission_profile(
    condition_id: str,
    profile: Mapping[str, object],
    *,
    initial_damage: tuple[float, float] = (0.0, 0.0),
) -> Trajectory:
    """Simulate a reusable-rocket duty cycle declared by an explicit profile.

    ``profile`` must contain exactly the eight mission fields used below. It
    describes preheat, ignition heating, hot dwell, and shutdown cooling, so
    experiment configurations can declare operating conditions without using
    this module's named-condition registry.
    """
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("condition_id must be a non-empty string")

    normalized_profile = _validate_mission_profile(profile)
    time = np.linspace(0.0, _MISSION_DURATION, _MISSION_SAMPLES)
    return simulate_condition(
        condition_id,
        time,
        _mission_forcing(time, **normalized_profile),
        initial_damage=initial_damage,
    )

def simulate_named_condition(condition_id: str) -> Trajectory:
    """Return one deterministic reusable-rocket mission cycle by identifier.

    The three legacy condition IDs are deliberately preserved. Their forcing
    traces now include preheat, rapid ignition heating, a modulated hot dwell,
    and shutdown cooling at representative material-wall temperatures.
    """
    try:
        profile = _MISSION_PROFILES[condition_id]
    except KeyError as error:
        choices = ", ".join(_MISSION_PROFILES)
        raise ValueError(f"unknown condition {condition_id!r}; choose from {choices}") from error
    return simulate_mission_profile(condition_id, profile)


def _mission_forcing(
    time: np.ndarray,
    *,
    baseline_temperature: float,
    peak_temperature: float,
    preheat_duration: float,
    ignition_duration: float,
    dwell_duration: float,
    oxygen_pressure: float,
    load: float,
    load_amplitude: float,
) -> dict[str, np.ndarray]:
    """Construct a four-phase material-wall duty cycle on ``time``."""
    preheat_end = preheat_duration
    ignition_end = preheat_end + ignition_duration
    dwell_end = ignition_end + dwell_duration
    if dwell_end >= time[-1]:
        raise ValueError("mission dwell must leave time for shutdown cooling")

    preheat_temperature = baseline_temperature + 160.0
    temperature = np.empty_like(time)
    preheat = time <= preheat_end
    ignition = (time > preheat_end) & (time <= ignition_end)
    dwell = (time > ignition_end) & (time <= dwell_end)
    cooling = time > dwell_end
    temperature[preheat] = baseline_temperature + (
        preheat_temperature - baseline_temperature
    ) * _smoothstep(time[preheat] / preheat_end)
    temperature[ignition] = preheat_temperature + (
        peak_temperature - preheat_temperature
    ) * _smoothstep((time[ignition] - preheat_end) / (ignition_end - preheat_end))
    dwell_phase = (time[dwell] - ignition_end) / dwell_duration
    temperature[dwell] = peak_temperature - 8.0 + 7.0 * np.sin(2.0 * np.pi * dwell_phase)
    cooling_start = peak_temperature - 8.0
    cooling_phase = (time[cooling] - dwell_end) / (time[-1] - dwell_end)
    temperature[cooling] = baseline_temperature + (
        cooling_start - baseline_temperature
    ) * np.exp(-2.8 * cooling_phase)

    sigma = np.full_like(time, load)
    sigma[dwell] = load + load_amplitude * np.sin(4.0 * np.pi * dwell_phase)
    sigma[ignition] = load + 0.25 * load_amplitude * _smoothstep(
        (time[ignition] - preheat_end) / (ignition_end - preheat_end)
    )
    oxygen = np.full_like(time, oxygen_pressure)
    oxygen[dwell] = oxygen_pressure * (1.0 + 0.06 * np.cos(2.0 * np.pi * dwell_phase))
    return {"T": temperature, "pO2": oxygen, "sigma": sigma}


def _validate_mission_profile(profile: Mapping[str, object]) -> dict[str, float]:
    """Validate and normalize the public reusable-rocket profile protocol."""
    if not isinstance(profile, Mapping):
        raise ValueError("mission profile must be a mapping")

    fields = set(profile)
    required = set(_MISSION_PROFILE_FIELDS)
    missing = required.difference(fields)
    unexpected = fields.difference(required)
    if missing:
        raise ValueError(
            f"mission profile is missing required fields: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise ValueError(
            f"mission profile contains unsupported fields: {', '.join(sorted(unexpected))}"
        )

    normalized = {}
    for field in _MISSION_PROFILE_FIELDS:
        value = profile[field]
        if isinstance(value, bool):
            raise ValueError(f"mission profile {field!r} must be a finite number")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"mission profile {field!r} must be a finite number"
            ) from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"mission profile {field!r} must be a finite number")
        normalized[field] = numeric_value

    baseline = normalized["baseline_temperature"]
    peak = normalized["peak_temperature"]
    preheat = normalized["preheat_duration"]
    ignition = normalized["ignition_duration"]
    dwell = normalized["dwell_duration"]
    oxygen = normalized["oxygen_pressure"]
    load = normalized["load"]
    amplitude = normalized["load_amplitude"]

    if baseline <= 0.0:
        raise ValueError("baseline_temperature must be positive")
    if not 900.0 <= peak <= 1450.0:
        raise ValueError("peak_temperature must be between 900 and 1450 K")
    if peak <= baseline + 160.0:
        raise ValueError(
            "peak_temperature must exceed baseline_temperature by more than 160 K"
        )
    if preheat <= 0.0:
        raise ValueError("preheat_duration must be positive")
    if ignition <= 0.0:
        raise ValueError("ignition_duration must be positive")
    if dwell <= 0.0:
        raise ValueError("dwell_duration must be positive")
    if preheat + ignition + dwell >= _MISSION_DURATION:
        raise ValueError("mission dwell must leave time for shutdown cooling")
    if oxygen < 0.0:
        raise ValueError("oxygen_pressure must be non-negative")
    if load < 0.0:
        raise ValueError("load must be non-negative")
    if amplitude < 0.0:
        raise ValueError("load_amplitude must be non-negative")
    if amplitude > load:
        raise ValueError("load_amplitude must not exceed load")
    return normalized

def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)

def _validate_initial_damage(initial_damage: object) -> np.ndarray:
    """Return a validated ``(Df, Do)`` state for a mission-cycle boundary."""
    try:
        raw_damage = np.asarray(initial_damage)
        damage = np.asarray(initial_damage, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "initial_damage must contain exactly two finite numbers"
        ) from error
    if raw_damage.dtype.kind == "b" or damage.shape != (2,):
        raise ValueError("initial_damage must contain exactly two finite numbers")
    if not np.all(np.isfinite(damage)):
        raise ValueError("initial_damage values must be finite")
    if np.any(damage < 0.0):
        raise ValueError("initial_damage values must be non-negative")
    if np.any(damage >= _DAMAGE_LIMIT):
        raise ValueError("initial_damage values must be below the damage limit")
    return np.array(damage, copy=True)

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


def oxidation_evaluation_truth() -> dict[str, object]:
    """Return hidden oxidation metadata for post-discovery evaluation only.

    Discovery modules do not import this function. Experiment evaluators call
    it only after model and parameter selection have completed.
    """
    return {
        "arrhenius_theta": _ACTIVATION_TEMPERATURE,
        "coefficient_by_factor": {
            "pO2": _OXIDATION_RATE,
            "pO2*sigma": _STRESS_OXIDATION_RATE,
            "pO2*Df": _FATIGUE_OXIDATION_RATE,
        },
    }


def simulate_realistic_condition(
    condition_id: str,
    profile: Mapping[str, object],
    *,
    initial_damage: tuple[float, float] = (0.0, 0.0),
    interaction_scale: float = 1.0,
) -> Trajectory:
    """Simulate one reusable-nozzle mission under the realistic oxidation regime.

    Uses an activation temperature ``Ea/R = 25000 K`` (Ea ~ 208 kJ/mol, within
    the Ni-superalloy oxidation range) so the Arrhenius factor is sharply
    temperature-activated and oxidation is confined to the hot dwell. The ODE is
    integrated with a fixed maximum step (see ``_REALISTIC_SOLVER_MAX_STEP``);
    without it the adaptive solver, seeing cold-phase rates ~1e-32, jumps across
    the active window in a single step and resolves ~zero oxidation damage.
    """
    if isinstance(interaction_scale, bool):
        raise ValueError("interaction_scale must be a finite non-negative number")
    try:
        interaction_multiplier = float(interaction_scale)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "interaction_scale must be a finite non-negative number"
        ) from error
    if not np.isfinite(interaction_multiplier) or interaction_multiplier < 0.0:
        raise ValueError("interaction_scale must be a finite non-negative number")

    normalized_profile = _validate_mission_profile(profile)
    time = np.linspace(0.0, _MISSION_DURATION, _MISSION_SAMPLES)
    forcing = _mission_forcing(time, **normalized_profile)
    initial_state = _validate_initial_damage(initial_damage)
    stress_reference = float(forcing["sigma"][0])

    def rhs(time_value: float, damage: np.ndarray) -> np.ndarray:
        temperature = float(np.interp(time_value, time, forcing["T"]))
        oxygen_pressure = float(np.interp(time_value, time, forcing["pO2"]))
        stress = float(np.interp(time_value, time, forcing["sigma"]))
        stress_range = abs(stress - stress_reference)
        fatigue_damage, oxidation_damage = damage
        thermal_factor = np.exp(-_REALISTIC_ACTIVATION_TEMPERATURE / temperature)
        fatigue_rate = (
            _REALISTIC_FATIGUE_RATE
            * stress_range**3
            / (1.0 - fatigue_damage)
        )
        oxidation_rate = thermal_factor * oxygen_pressure * (
            _REALISTIC_OXIDATION_C0
            + _REALISTIC_OXIDATION_C1 * stress
            + interaction_multiplier * _REALISTIC_OXIDATION_C2 * fatigue_damage
        )
        return np.array([fatigue_rate, oxidation_rate])

    solution = solve_ivp(
        rhs,
        (time[0], time[-1]),
        initial_state,
        t_eval=time,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        max_step=_REALISTIC_SOLVER_MAX_STEP,
    )
    if not solution.success:
        raise RuntimeError(
            f"realistic aging simulation failed: {solution.message}"
        )
    if np.any(solution.y < 0.0) or np.any(solution.y >= _DAMAGE_LIMIT):
        raise ValueError("forcing produces damage outside the supported range")

    return Trajectory(
        t=time,
        states={**forcing, "Df": solution.y[0], "Do": solution.y[1]},
        condition_id=condition_id,
    )


def realistic_oxidation_evaluation_truth() -> dict[str, object]:
    """Return hidden oxidation metadata for the realistic regime.

    Discovery modules do not import this function. Experiment evaluators call
    it only after model and parameter selection have completed.
    """
    return {
        "arrhenius_theta": _REALISTIC_ACTIVATION_TEMPERATURE,
        "coefficient_by_factor": {
            "pO2": _REALISTIC_OXIDATION_C0,
            "pO2*sigma": _REALISTIC_OXIDATION_C1,
            "pO2*Df": _REALISTIC_OXIDATION_C2,
        },
    }
