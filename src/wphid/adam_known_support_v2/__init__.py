"""Conditional Adam comparator with the same supplied active-support prior."""

from .adapter import KnownSupportBudget, fit_known_support

__all__ = ['KnownSupportBudget', 'fit_known_support']
