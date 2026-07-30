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
    coefficient_dimension: tuple[int, ...] | None = None
    canonical_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("candidate name must be a non-empty string")
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("candidate group must be a non-empty string")
        dimension = tuple(self.dimension)
        parents = tuple(self.parents)
        coefficient_dimension = (
            None
            if self.coefficient_dimension is None
            else tuple(self.coefficient_dimension)
        )
        if (
            coefficient_dimension is not None
            and len(coefficient_dimension) != len(dimension)
        ):
            raise ValueError(
                "candidate and coefficient dimensions must use the same basis"
            )
        identity = (
            self.name
            if self.canonical_identity is None
            else self.canonical_identity
        )
        if not isinstance(identity, str) or not identity:
            raise ValueError("canonical_identity must be a non-empty string")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "coefficient_dimension", coefficient_dimension)
        object.__setattr__(self, "canonical_identity", identity)


@dataclass(frozen=True)
class DiscoveryResult:
    """The selected candidate support and validation diagnostics."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    support: np.ndarray
    validation_error: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")

        coefficients = np.array(self.coefficients, dtype=float, copy=True)
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be one-dimensional")
        if coefficients.size != len(names):
            raise ValueError("DiscoveryResult requires one coefficient per name")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must contain only finite values")

        support = np.array(self.support, copy=True)
        if support.ndim != 1:
            raise ValueError("support must be one-dimensional")
        if (
            np.issubdtype(support.dtype, np.bool_)
            or not np.issubdtype(support.dtype, np.integer)
        ):
            raise ValueError("support must contain integer indices")
        support = support.astype(np.intp, copy=False)
        if support.size and (
            np.any(support < 0) or np.any(support >= len(names))
        ):
            raise ValueError("support indices must be within names")
        if support.size > 1 and not np.all(np.diff(support) > 0):
            raise ValueError("support indices must be strictly increasing")

        try:
            validation_error = float(self.validation_error)
        except (TypeError, ValueError) as error:
            raise ValueError("validation_error must be a finite number") from error
        if not np.isfinite(validation_error):
            raise ValueError("validation_error must be finite")
        if validation_error < 0.0:
            raise ValueError("validation_error must be non-negative")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

        coefficients.setflags(write=False)
        support.setflags(write=False)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "validation_error", validation_error)
        object.__setattr__(self, "diagnostics", _immutable_mapping(self.diagnostics))
