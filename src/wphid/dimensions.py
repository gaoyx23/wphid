"""Equation-level dimensional consistency helpers."""

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from wphid.types import Candidate


Dimension: TypeAlias = tuple[int, ...]
DAMAGE_RATE: Dimension = (0, 0, -1, 0)


def normalize_dimension(value: Iterable[int], label: str = "dimension") -> Dimension:
    """Return an integer exponent tuple suitable for dimensional arithmetic."""
    dimension = tuple(value)
    if not dimension:
        raise ValueError(f"{label} must contain at least one base-unit exponent")
    if any(
        isinstance(exponent, (bool, np.bool_))
        or not isinstance(exponent, (int, np.integer))
        for exponent in dimension
    ):
        raise ValueError(f"{label} must contain only integer exponents")
    return tuple(int(exponent) for exponent in dimension)


def add_dimensions(left: Iterable[int], right: Iterable[int]) -> Dimension:
    """Multiply quantities by adding their base-unit exponents."""
    left_dimension = normalize_dimension(left, "left dimension")
    right_dimension = normalize_dimension(right, "right dimension")
    if len(left_dimension) != len(right_dimension):
        raise ValueError("dimensions must use the same base-unit basis")
    return tuple(
        left_exponent + right_exponent
        for left_exponent, right_exponent in zip(
            left_dimension, right_dimension, strict=True
        )
    )


def subtract_dimensions(left: Iterable[int], right: Iterable[int]) -> Dimension:
    """Divide quantities by subtracting their base-unit exponents."""
    left_dimension = normalize_dimension(left, "left dimension")
    right_dimension = normalize_dimension(right, "right dimension")
    if len(left_dimension) != len(right_dimension):
        raise ValueError("dimensions must use the same base-unit basis")
    return tuple(
        left_exponent - right_exponent
        for left_exponent, right_exponent in zip(
            left_dimension, right_dimension, strict=True
        )
    )


def validate_equation_dimensions(
    candidates: Iterable[Candidate], target_dimension: Iterable[int]
) -> Dimension:
    """Require every coefficient-candidate product to match one target."""
    target = normalize_dimension(target_dimension, "target dimension")
    for candidate in candidates:
        feature_dimension = normalize_dimension(
            candidate.dimension,
            f"candidate {candidate.name!r} dimension",
        )
        coefficient_dimension = candidate.coefficient_dimension
        if coefficient_dimension is None:
            raise ValueError(
                f"candidate {candidate.name!r} must declare a coefficient dimension"
            )
        coefficient_dimension = normalize_dimension(
            coefficient_dimension,
            f"candidate {candidate.name!r} coefficient dimension",
        )
        if len(feature_dimension) != len(target):
            raise ValueError(
                f"candidate {candidate.name!r} dimension basis does not match target"
            )
        product_dimension = add_dimensions(
            feature_dimension, coefficient_dimension
        )
        if product_dimension != target:
            raise ValueError(
                f"candidate {candidate.name!r} times its coefficient has dimension "
                f"{product_dimension}, which does not match target {target}"
            )
    return target
