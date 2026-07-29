import numpy as np
import pytest

from wphid.sparse.stlsq import fit_stlsq


def test_fit_stlsq_recovers_support_and_physical_coefficients_across_column_scales():
    """Removing standard-deviation scaling would drop the small raw column."""
    rng = np.random.default_rng(14)
    A = np.column_stack(
        (
            1e-4 * rng.normal(size=400),
            1e4 * rng.normal(size=400),
            1e2 * rng.normal(size=400),
            1e-2 * rng.normal(size=400),
        )
    )
    expected = np.array([5_000.0, 0.0, -0.025, 0.0])
    b = A @ expected

    result = fit_stlsq(A, b, threshold=0.1)

    np.testing.assert_array_equal(result.support, np.array([0, 2]))
    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-8, atol=1e-12)
    assert result.validation_error == pytest.approx(0.0, abs=1e-12)


def test_fit_stlsq_excludes_constant_columns_and_reports_conditioning():
    """Treating a constant column as scalable would divide by zero."""
    x = np.linspace(-2.0, 3.0, 80)
    A = np.column_stack((x, np.full_like(x, 7.0), x**2))
    b = 1.5 * x - 0.25 * x**2

    result = fit_stlsq(A, b, threshold=0.05)

    np.testing.assert_array_equal(result.support, np.array([0, 2]))
    assert result.coefficients[1] == 0.0
    np.testing.assert_array_equal(
        result.diagnostics["excluded_zero_variance_columns"], np.array([1])
    )
    assert np.isfinite(result.diagnostics["condition_number"])
    assert np.isfinite(result.diagnostics["scaled_condition_number"])
