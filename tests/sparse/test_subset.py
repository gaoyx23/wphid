import numpy as np
import pytest

from wphid.sparse.subset import best_subset_aic, best_subset_bic


def _orthonormal_design(n, p, seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, p))
    q, _ = np.linalg.qr(raw)
    return q


def _null_space_noise(A, n, seed, sigma):
    """Isotropic noise projected onto A's null space. Its projection on every
    column is exactly zero, so a spurious column can never reduce RSS and AIC
    rejects it deterministically (with isotropic noise the acceptance of a
    spurious column is a ~16% coin flip per column, which would flake the test)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(A)
    eps = rng.normal(size=n)
    eps = eps - q @ (q.T @ eps)
    return sigma * eps


def test_best_subset_aic_recovers_sparse_support_under_noise():
    """AIC must recover the true support when the residual sits at the noise
    floor (not the floating-point floor, where n*log(RSS) over-fits)."""
    A = _orthonormal_design(500, 5, seed=5)
    true_support = np.array([1, 3])
    b = A[:, true_support] @ np.array([2.0, -1.5]) + _null_space_noise(A, 500, 17, 0.1)

    support, score = best_subset_aic(A, b)

    np.testing.assert_array_equal(support, true_support)
    assert np.isfinite(score)


def test_best_subset_aic_prefers_true_support_over_full_model_under_noise():
    """AIC must balance fit against size: under noise it rejects both the
    saturated model and a model missing a true term."""
    A = _orthonormal_design(400, 6, seed=11)
    true_support = np.array([0, 2, 4])
    b = A[:, true_support] @ np.array([1.0, -2.0, 0.5]) + _null_space_noise(A, 400, 19, 0.05)

    support, score = best_subset_aic(A, b)

    np.testing.assert_array_equal(support, true_support)
    assert np.isfinite(score)


def test_best_subset_aic_is_deterministic():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(200, 4))
    b = A @ np.array([0.0, 1.5, 0.0, -0.5]) + 0.02 * rng.normal(size=200)

    first = best_subset_aic(A, b)
    second = best_subset_aic(A, b)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_best_subset_aic_skips_rank_deficient_subsets():
    """Duplicate columns must never both enter the support; the tie-break then
    selects the lowest-index singleton."""
    rng = np.random.default_rng(9)
    base = rng.normal(size=(300, 1))
    A = np.column_stack((base, base, rng.normal(size=(300, 1))))
    b = base[:, 0] * 3.0 + 0.05 * rng.normal(size=300)

    support, score = best_subset_aic(A, b)

    assert support.size == 1
    assert support[0] == 0
    assert np.isfinite(score)


def test_best_subset_aic_rejects_oversized_library():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(100, 21))
    b = rng.normal(size=100)

    with pytest.raises(ValueError, match="columns"):
        best_subset_aic(A, b)


def test_best_subset_bic_recovers_sparse_support_under_noise():
    A = _orthonormal_design(500, 5, seed=23)
    expected = np.array([1, 4])
    b = A[:, expected] @ np.array([2.0, -1.0]) + _null_space_noise(A, 500, 29, 0.1)
    support, score = best_subset_bic(A, b)
    np.testing.assert_array_equal(support, expected)
    assert np.isfinite(score)


def test_best_subset_bic_is_deterministic_and_skips_rank_deficiency():
    rng = np.random.default_rng(31)
    base = rng.normal(size=(300, 1))
    A = np.column_stack((base, base, rng.normal(size=(300, 1))))
    b = 3.0 * base[:, 0] + 0.05 * rng.normal(size=300)
    first = best_subset_bic(A, b)
    second = best_subset_bic(A, b)
    assert 0 in first[0]
    assert 1 not in first[0]
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_best_subset_bic_rejects_oversized_library():
    rng = np.random.default_rng(37)
    with pytest.raises(ValueError, match="columns"):
        best_subset_bic(rng.normal(size=(100, 21)), rng.normal(size=100))
