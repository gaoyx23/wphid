"""Deterministic observation-only trajectories for WPHID benchmarks."""

from .aging_ode import simulate_condition, simulate_named_condition

__all__ = ("simulate_condition", "simulate_named_condition")
