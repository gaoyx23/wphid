"""Train-only aggregation of bootstrap support evidence."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _readonly(values: np.ndarray, *, dtype: type[float] | None = float) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StabilitySummary:
    """Support evidence accumulated over bootstrap fits of training data."""

    names: tuple[str, ...]
    inclusion_probabilities: np.ndarray
    sign_consistency: np.ndarray
    support: np.ndarray
    parameter_selection_frequencies: Mapping[float, float]
    n_resamples: int
    inclusion_threshold: float
    sign_consistency_threshold: float
    selection_tolerance: float

    def __post_init__(self) -> None:
        names = tuple(self.names)
        inclusion = _readonly(self.inclusion_probabilities)
        signs = _readonly(self.sign_consistency)
        support = _readonly(self.support, dtype=int)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")
        if (
            inclusion.ndim != 1
            or signs.ndim != 1
            or inclusion.size != len(names)
            or signs.size != len(names)
        ):
            raise ValueError("stability vectors must align with names")
        if (
            not np.isfinite(inclusion).all()
            or not np.isfinite(signs).all()
            or np.any(inclusion < 0.0)
            or np.any(inclusion > 1.0)
            or np.any(signs < 0.0)
            or np.any(signs > 1.0)
        ):
            raise ValueError(
                "stability vectors must contain finite probabilities"
            )
        if support.ndim != 1 or (
            support.size
            and (np.any(support < 0) or np.any(support >= len(names)))
        ):
            raise ValueError("support indices must align with names")
        if support.size > 1 and not np.all(np.diff(support) > 0):
            raise ValueError("support must be strictly increasing")
        if not isinstance(self.n_resamples, (int, np.integer)) or self.n_resamples < 1:
            raise ValueError("n_resamples must be a positive integer")
        thresholds = (
            self.inclusion_threshold,
            self.sign_consistency_threshold,
            self.selection_tolerance,
        )
        if not all(np.isfinite(value) for value in thresholds):
            raise ValueError("stability thresholds must be finite")
        if not 0.0 <= self.inclusion_threshold <= 1.0:
            raise ValueError("inclusion_threshold must be between zero and one")
        if not 0.0 <= self.sign_consistency_threshold <= 1.0:
            raise ValueError(
                "sign_consistency_threshold must be between zero and one"
            )
        if self.selection_tolerance < 0.0:
            raise ValueError("selection_tolerance must be non-negative")
        try:
            frequencies = {
                float(key): float(value)
                for key, value in self.parameter_selection_frequencies.items()
            }
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "parameter selections must map finite values to probabilities"
            ) from error
        if any(
            not np.isfinite(key)
            or not np.isfinite(value)
            or value < 0.0
            or value > 1.0
            for key, value in frequencies.items()
        ):
            raise ValueError(
                "parameter selections must map finite values to probabilities"
            )
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "inclusion_probabilities", inclusion)
        object.__setattr__(self, "sign_consistency", signs)
        object.__setattr__(self, "support", support)
        object.__setattr__(
            self,
            "parameter_selection_frequencies",
            MappingProxyType(frequencies),
        )


def aggregate_stability(
    coefficients: np.ndarray,
    *,
    candidate_names: Sequence[str] | None = None,
    inclusion_threshold: float = 0.8,
    sign_consistency_threshold: float = 0.8,
    selection_tolerance: float = 0.0,
    theta_values: Sequence[float | None] | None = None,
) -> StabilitySummary:
    """Aggregate bootstrap coefficients without accessing truth or test data."""
    fitted = np.asarray(coefficients, dtype=float)
    if fitted.ndim != 2 or fitted.shape[0] < 1 or fitted.shape[1] < 1:
        raise ValueError(
            "coefficients must have shape (n_resamples, n_candidates)"
        )
    if not np.isfinite(fitted).all():
        raise ValueError("coefficients must contain only finite values")
    if not 0.0 <= inclusion_threshold <= 1.0:
        raise ValueError("inclusion_threshold must be between zero and one")
    if not 0.0 <= sign_consistency_threshold <= 1.0:
        raise ValueError("sign_consistency_threshold must be between zero and one")
    if not np.isfinite(selection_tolerance) or selection_tolerance < 0.0:
        raise ValueError("selection_tolerance must be finite and non-negative")
    if candidate_names is None:
        names = tuple(f"term_{index}" for index in range(fitted.shape[1]))
    else:
        names = tuple(candidate_names)
    if len(names) != fitted.shape[1]:
        raise ValueError("candidate_names must align with coefficient columns")

    selected = np.abs(fitted) > selection_tolerance
    inclusion = np.mean(selected, axis=0)
    positive = np.count_nonzero(fitted > selection_tolerance, axis=0)
    negative = np.count_nonzero(fitted < -selection_tolerance, axis=0)
    selected_count = np.count_nonzero(selected, axis=0)
    signs = np.divide(
        np.maximum(positive, negative),
        selected_count,
        out=np.zeros(fitted.shape[1], dtype=float),
        where=selected_count > 0,
    )
    support = np.flatnonzero(
        (inclusion >= inclusion_threshold)
        & (signs >= sign_consistency_threshold)
    )

    frequencies: dict[float, float] = {}
    if theta_values is not None:
        if len(theta_values) != fitted.shape[0]:
            raise ValueError("theta_values must provide one value per resample")
        selected_thetas = [float(theta) for theta in theta_values if theta is not None]
        if not all(np.isfinite(theta) for theta in selected_thetas):
            raise ValueError("theta_values must be finite when supplied")
        if selected_thetas:
            values, counts = np.unique(selected_thetas, return_counts=True)
            frequencies = {
                float(value): float(count / len(selected_thetas))
                for value, count in zip(values, counts, strict=True)
            }
    return StabilitySummary(
        names=names,
        inclusion_probabilities=inclusion,
        sign_consistency=signs,
        support=support,
        parameter_selection_frequencies=frequencies,
        n_resamples=fitted.shape[0],
        inclusion_threshold=float(inclusion_threshold),
        sign_consistency_threshold=float(sign_consistency_threshold),
        selection_tolerance=float(selection_tolerance),
    )