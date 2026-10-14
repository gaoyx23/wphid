"""Bootstrap orchestration for training-only sparse stability selection."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from wphid.stability.aggregate import StabilitySummary, aggregate_stability
from wphid.stability.bootstrap import WeakFormInput, bootstrap_grouped_weak_form


SelectorResult = np.ndarray | tuple[np.ndarray, float | None]
SparseSelector = Callable[
    [np.ndarray, np.ndarray, int, np.random.Generator], SelectorResult
]


def run_stability_selection(
    conditions: Mapping[str, WeakFormInput],
    *,
    candidate_names: Sequence[str],
    selector: SparseSelector,
    n_resamples: int,
    block_length: int,
    random_seed: int | np.integer,
    inclusion_threshold: float = 0.8,
    sign_consistency_threshold: float = 0.8,
    selection_tolerance: float = 0.0,
) -> StabilitySummary:
    """Rerun ``selector`` on deterministic grouped block-bootstrap samples.

    This API accepts only weak-form training matrices and targets. It has no
    parameter for truth labels, validation conditions, or test conditions.
    Each replicate passes a deterministic seed and fresh RNG to the selector.
    """
    if not callable(selector):
        raise ValueError("selector must be callable")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(random_seed, (int, np.integer)):
        raise ValueError("random_seed must be an integer")

    names = tuple(candidate_names)
    rng = np.random.default_rng(random_seed)
    coefficients = []
    theta_values = []
    for _ in range(int(n_resamples)):
        bootstrap_seed = int(
            rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        )
        selector_seed = int(
            rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        )
        sample = bootstrap_grouped_weak_form(
            conditions,
            block_length=block_length,
            random_seed=bootstrap_seed,
        )
        matrix, target = sample.stacked()
        selector_rng = np.random.default_rng(selector_seed)
        result = selector(matrix, target, selector_seed, selector_rng)
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("selector tuple result must be (coefficients, theta)")
            fitted, theta = result
        else:
            fitted, theta = result, None
        fitted = np.asarray(fitted, dtype=float)
        if fitted.ndim != 1 or fitted.size != len(names):
            raise ValueError("selector coefficients must align with candidate_names")
        if not np.isfinite(fitted).all():
            raise ValueError("selector coefficients must be finite")
        if theta is not None and not np.isfinite(float(theta)):
            raise ValueError("selector theta must be finite when supplied")
        coefficients.append(fitted)
        theta_values.append(None if theta is None else float(theta))
    return aggregate_stability(
        np.asarray(coefficients),
        candidate_names=names,
        inclusion_threshold=inclusion_threshold,
        sign_consistency_threshold=sign_consistency_threshold,
        selection_tolerance=selection_tolerance,
        theta_values=theta_values,
    )