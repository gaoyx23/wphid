"""Deterministic grouped, contiguous-block resampling for weak-form rows."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


def _readonly_array(values: np.ndarray, *, dtype: type[float] | None = float) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class WeakFormCondition:
    """One training condition's aligned weak-form regression rows."""

    condition_id: str
    matrix: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id:
            raise ValueError("condition_id must be a non-empty string")
        matrix = _readonly_array(self.matrix)
        target = _readonly_array(self.target)
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError("matrix must be two-dimensional with at least one row")
        if target.ndim != 1:
            raise ValueError("target must be one-dimensional")
        if matrix.shape[0] != target.size:
            raise ValueError("matrix and target must have the same number of rows")
        if not np.isfinite(matrix).all() or not np.isfinite(target).all():
            raise ValueError("matrix and target must contain only finite values")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "target", target)


WeakFormInput: TypeAlias = tuple[np.ndarray, np.ndarray] | WeakFormCondition


@dataclass(frozen=True)
class BootstrappedCondition:
    """A sampled complete condition with its within-condition window indices."""

    condition_id: str
    source_condition_id: str
    matrix: np.ndarray
    target: np.ndarray
    row_indices: np.ndarray
    block_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id:
            raise ValueError("condition_id must be a non-empty string")
        if not isinstance(self.source_condition_id, str) or not self.source_condition_id:
            raise ValueError("source_condition_id must be a non-empty string")
        matrix = _readonly_array(self.matrix)
        target = _readonly_array(self.target)
        raw_rows = np.asarray(self.row_indices)
        if (
            np.issubdtype(raw_rows.dtype, np.bool_)
            or not np.issubdtype(raw_rows.dtype, np.integer)
        ):
            raise ValueError("row_indices must contain integers")
        rows = _readonly_array(raw_rows, dtype=int)
        lengths = tuple(self.block_lengths)
        if matrix.ndim != 2 or target.ndim != 1 or rows.ndim != 1:
            raise ValueError("sampled matrix, target, and row_indices have invalid shapes")
        if matrix.shape[0] != target.size or target.size != rows.size:
            raise ValueError("sampled matrix, target, and row_indices must align")
        if not lengths or any(
            not isinstance(length, (int, np.integer)) or length < 2
            for length in lengths
        ):
            raise ValueError("block_lengths must contain integers of at least two")
        if sum(lengths) != rows.size:
            raise ValueError("block_lengths must sum to the sampled row count")
        if not np.isfinite(matrix).all() or not np.isfinite(target).all():
            raise ValueError("sampled matrix and target must contain only finite values")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "block_lengths", lengths)


@dataclass(frozen=True)
class GroupedBootstrapSample:
    """A grouped bootstrap replicate ready for a sparse weak-form fit."""

    conditions: tuple[BootstrappedCondition, ...]

    def __post_init__(self) -> None:
        conditions = tuple(self.conditions)
        if not conditions:
            raise ValueError("a bootstrap sample requires at least one condition")
        widths = {condition.matrix.shape[1] for condition in conditions}
        if len(widths) != 1:
            raise ValueError("sampled conditions must have the same number of columns")
        identifiers = tuple(condition.condition_id for condition in conditions)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("sampled condition_ids must be unique")
        object.__setattr__(self, "conditions", conditions)

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(condition.condition_id for condition in self.conditions)

    def stacked(self) -> tuple[np.ndarray, np.ndarray]:
        """Return rows stacked by sampled condition, preserving group boundaries."""
        return (
            np.vstack([condition.matrix for condition in self.conditions]),
            np.concatenate([condition.target for condition in self.conditions]),
        )


def _as_conditions(
    conditions: Mapping[str, WeakFormInput],
) -> tuple[WeakFormCondition, ...]:
    if not isinstance(conditions, Mapping) or not conditions:
        raise ValueError("conditions must be a non-empty mapping")
    converted = []
    for condition_id, values in conditions.items():
        if isinstance(values, WeakFormCondition):
            if values.condition_id != condition_id:
                raise ValueError("mapping key must match WeakFormCondition.condition_id")
            converted.append(values)
        else:
            try:
                matrix, target = values
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "each condition must be a (matrix, target) pair"
                ) from error
            converted.append(WeakFormCondition(condition_id, matrix, target))
    converted.sort(key=lambda condition: condition.condition_id)
    widths = {condition.matrix.shape[1] for condition in converted}
    if len(widths) != 1:
        raise ValueError("conditions must have the same number of columns")
    return tuple(converted)


def _block_indices(
    n_rows: int, block_length: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, ...]]:
    n_blocks = (n_rows + block_length - 1) // block_length
    starts = rng.integers(
        0, n_rows - block_length + 1, size=n_blocks
    )
    rows = [
        np.arange(int(start), int(start) + block_length, dtype=int)
        for start in starts
    ]
    return np.concatenate(rows), (block_length,) * n_blocks


def bootstrap_grouped_weak_form(
    conditions: Mapping[str, WeakFormInput],
    *,
    block_length: int,
    random_seed: int | np.integer | None,
    n_conditions: int | None = None,
) -> GroupedBootstrapSample:
    """Sample training conditions and contiguous weak-form window blocks.

    Conditions are drawn with replacement as complete groups. Within each
    chosen group, fixed-length contiguous blocks are drawn with replacement.
    A full final block pads nondivisible condition lengths.
    """
    source_conditions = _as_conditions(conditions)
    if not isinstance(block_length, (int, np.integer)) or block_length < 2:
        raise ValueError("block_length must be an integer of at least two")
    if any(block_length > condition.target.size for condition in source_conditions):
        raise ValueError("block_length must not exceed any condition row count")
    if n_conditions is None:
        n_conditions = len(source_conditions)
    if not isinstance(n_conditions, (int, np.integer)) or n_conditions < 1:
        raise ValueError("n_conditions must be a positive integer")
    if random_seed is not None and not isinstance(random_seed, (int, np.integer)):
        raise ValueError("random_seed must be an integer or None")

    rng = np.random.default_rng(random_seed)
    selected = rng.integers(0, len(source_conditions), size=int(n_conditions))
    bootstrapped = []
    for draw_index, source_index in enumerate(selected):
        source = source_conditions[int(source_index)]
        row_indices, lengths = _block_indices(
            source.target.size, int(block_length), rng
        )
        bootstrapped.append(
            BootstrappedCondition(
                condition_id=f"bootstrap_{draw_index}",
                source_condition_id=source.condition_id,
                matrix=source.matrix[row_indices],
                target=source.target[row_indices],
                row_indices=row_indices,
                block_lengths=lengths,
            )
        )
    return GroupedBootstrapSample(tuple(bootstrapped))