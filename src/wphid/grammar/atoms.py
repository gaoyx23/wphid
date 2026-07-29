"""Declared physical atoms for the aging interaction grammar."""

from collections.abc import Mapping

import numpy as np

from wphid.types import Candidate


# Dimensions use exponents of (mass, length, time, temperature).
DIMENSIONLESS = (0, 0, 0, 0)
PRESSURE = (1, -1, -2, 0)


def _state(name: str):
    def evaluate(states: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(states[name], dtype=float)

    return evaluate


def _arrhenius_candidate(arrhenius_temperature: float) -> Candidate:
    parameter = float(arrhenius_temperature)
    if not np.isfinite(parameter) or parameter <= 0.0:
        raise ValueError("arrhenius_temperature must be finite and positive")
    name = f"arr(T;theta={parameter:g})"

    def evaluate(states: Mapping[str, np.ndarray]) -> np.ndarray:
        temperature = np.asarray(states["T"], dtype=float)
        if np.any(temperature <= 0):
            raise ValueError(f"{name} requires positive absolute temperature")
        return np.exp(-parameter / temperature)

    return Candidate(name, evaluate, DIMENSIONLESS, "F", (name,))


def _saturation(name: str):
    def evaluate(states: Mapping[str, np.ndarray]) -> np.ndarray:
        damage = np.asarray(states[name], dtype=float)
        if np.any(damage >= 1.0):
            raise ValueError(f"sat({name}) requires {name} < 1")
        return 1.0 / (1.0 - damage)

    return evaluate


def aging_atoms(arrhenius_temperature: float = 1000.0) -> list[Candidate]:
    """Return the declared, deterministic atom library for aging models.

    Temperature enters only through the dimensionless Arrhenius modulation;
    `sat(Df)` is the explicitly declared fatigue saturation transform.
    """

    return [
        _arrhenius_candidate(arrhenius_temperature),
        Candidate("pO2", _state("pO2"), PRESSURE, "F", ("pO2",)),
        Candidate("sigma", _state("sigma"), PRESSURE, "F", ("sigma",)),
        Candidate("Df", _state("Df"), DIMENSIONLESS, "F", ("Df",)),
        Candidate("sat(Df)", _saturation("Df"), DIMENSIONLESS, "F", ("sat(Df)",)),
    ]
