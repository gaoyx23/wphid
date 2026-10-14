"""Sparse-regression routines for weak-form discovery."""

from wphid.sparse.stlsq import fit_stlsq
from wphid.sparse.subset import best_subset_aic, best_subset_bic

__all__ = ["fit_stlsq", "best_subset_aic", "best_subset_bic"]
