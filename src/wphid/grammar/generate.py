"""Bounded, deterministic composition of dimensionally compatible atoms."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from wphid.types import Candidate


_MAX_SUPPORTED_ORDER = 3
_DISPLAY_ORDER = {"pO2": 1, "sigma": 2, "Df": 3, "sat(Df)": 4}


@dataclass(frozen=True)
class _Node:
    candidate: Candidate
    canonical_key: tuple[str, ...]


def _canonical_key(candidate: Candidate) -> tuple[str, ...]:
    """Identify an expression by canonical physics identity, not display name."""
    return (candidate.canonical_identity,)


def _add_dimensions(dimensions: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    first = dimensions[0]
    if any(len(dimension) != len(first) for dimension in dimensions):
        raise ValueError("all candidate dimensions must have the same basis")
    return tuple(sum(exponents) for exponents in zip(*dimensions, strict=True))


def _is_arrhenius(candidate: Candidate) -> bool:
    return candidate.name.startswith("arr(T;theta=")


def _display_key(node: _Node) -> tuple[int, str, tuple[str, ...]]:
    rank = 0 if _is_arrhenius(node.candidate) else _DISPLAY_ORDER.get(
        node.candidate.name, len(_DISPLAY_ORDER)
    )
    return (rank, node.candidate.name, node.canonical_key)


def _product_evaluator(nodes: tuple[_Node, ...]):
    def evaluate(states: Mapping[str, np.ndarray]) -> np.ndarray:
        values = [
            np.asarray(node.candidate.evaluate(states), dtype=float) for node in nodes
        ]
        return np.prod(values, axis=0)

    return evaluate


def _is_admissible(nodes: tuple[_Node, ...]) -> bool:
    # An Arrhenius modulation is a single temperature-dependent factor, never
    # an arbitrary repeated nonlinear transformation.
    return sum(_is_arrhenius(node.candidate) for node in nodes) <= 1


def _group(order: int) -> str:
    return "F" if order == 1 else f"G{order}"


def generate_candidates(
    atoms: Iterable[Candidate],
    target_dimension: tuple[int, ...],
    max_order: int = _MAX_SUPPORTED_ORDER,
    coefficient_dimension: tuple[int, ...] | None = None,
) -> list[Candidate]:
    """Generate admissible products up to ``max_order`` at one exact dimension.

    Products are de-duplicated using sorted canonical leaf trees before their
    human-readable names are constructed, making commutative composition
    deterministic regardless of enumeration order.
    """

    if not 1 <= max_order <= _MAX_SUPPORTED_ORDER:
        raise ValueError(f"max_order must be between 1 and {_MAX_SUPPORTED_ORDER}")

    nodes = sorted(
        (_Node(candidate, _canonical_key(candidate)) for candidate in atoms),
        key=lambda node: (
            node.canonical_key,
            node.candidate.dimension,
            node.candidate.name,
        ),
    )
    if nodes and any(
        len(node.candidate.dimension) != len(target_dimension) for node in nodes
    ):
        raise ValueError("target_dimension must use the candidates' dimension basis")
    if (
        coefficient_dimension is not None
        and len(coefficient_dimension) != len(target_dimension)
    ):
        raise ValueError("coefficient_dimension must use the target dimension basis")

    generated: dict[tuple[tuple[str, ...], ...], Candidate] = {}
    for order in range(1, max_order + 1):
        for selection in combinations_with_replacement(nodes, order):
            if not _is_admissible(selection):
                continue
            dimensions = [node.candidate.dimension for node in selection]
            if _add_dimensions(dimensions) != tuple(target_dimension):
                continue

            tree = tuple(sorted((node.canonical_key for node in selection)))
            if tree in generated:
                continue

            display_nodes = tuple(sorted(selection, key=_display_key))
            parents = tuple(node.candidate.name for node in display_nodes)
            canonical_identity = "product(" + ",".join(
                identity
                for key in tree
                for identity in key
            ) + ")"
            generated[tree] = Candidate(
                name="*".join(parents),
                evaluate=_product_evaluator(display_nodes),
                dimension=tuple(target_dimension),
                group=_group(order),
                parents=parents,
                coefficient_dimension=coefficient_dimension,
                canonical_identity=canonical_identity,
            )

    return sorted(
        generated.values(),
        key=lambda candidate: (
            len(candidate.parents),
            candidate.name,
            candidate.parents,
        ),
    )
