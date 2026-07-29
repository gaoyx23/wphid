"""Immutable data contracts shared by truth generation and discovery code."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


def _readonly_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


def _immutable_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType({key: _freeze(value) for key, value in values.items()})


@dataclass(frozen=True)
class Trajectory:
    """A time-indexed collection of observed state arrays."""

    t: np.ndarray
    states: Mapping[str, np.ndarray]
    condition_id: str

    def __post_init__(self) -> None:
        time = _readonly_array(self.t)
        if time.ndim != 1:
            raise ValueError("time axis must be one-dimensional")
        if time.size > 1 and not np.all(np.diff(time) > 0):
            raise ValueError("time axis must be strictly increasing")

        states = {}
        for name, state in self.states.items():
            values = _readonly_array(state)
            if values.ndim == 0 or values.shape[0] != time.shape[0]:
                raise ValueError(f"state {name!r} must match the time axis")
            states[name] = values

        object.__setattr__(self, "t", time)
        object.__setattr__(self, "states", _immutable_mapping(states))


@dataclass(frozen=True)
class Candidate:
    """A named admissible expression that evaluates over state arrays."""

    name: str
    evaluate: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    dimension: tuple[int, ...]
    group: str
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", tuple(self.dimension))
        object.__setattr__(self, "parents", tuple(self.parents))


@dataclass(frozen=True)
class DiscoveryResult:
    """The selected candidate support and validation diagnostics."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    support: np.ndarray
    validation_error: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "coefficients", _readonly_array(self.coefficients))
        object.__setattr__(self, "support", _readonly_array(self.support))
        object.__setattr__(self, "diagnostics", _immutable_mapping(self.diagnostics))
