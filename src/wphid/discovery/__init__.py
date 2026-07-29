"""Residual-driven, physics-grammar expansion."""

from wphid.discovery.residual import (
    ExpansionScore,
    ResidualBlock,
    discover_with_expansion,
    rank_expansions,
)

__all__ = [
    "ExpansionScore",
    "ResidualBlock",
    "discover_with_expansion",
    "rank_expansions",
]
