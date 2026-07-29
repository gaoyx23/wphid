"""Immutable data contracts shared by truth generation and discovery code."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """A time-indexed collection of observed state arrays."""

    t: np.ndarray
    states: Mapping[str, np.ndarray]
    condition_id: str

    def __post_init__(self) -> None:
        time = np.asarray(self.t)
        if time.ndim != 1:
            raise ValueError("time axis must be one-dimensional")
        if time.size > 1 and not np.all(np.diff(time) > 0):
            raise ValueError("time axis must be strictly increasing")

        for name, state in self.states.items():
            values = np.asarray(state)
            if values.ndim == 0 or values.shape[0] != time.shape[0]:
                raise ValueError(f"state {name!r} must match the time axis")


@dataclass(frozen=True)
class Candidate:
    """A named admissible expression that evaluates over state arrays."""

    name: str
    evaluate: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    dimension: tuple[int, ...]
    group: str
    parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    """The selected candidate support and validation diagnostics."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    support: np.ndarray
    validation_error: float
    diagnostics: Mapping[str, Any]
