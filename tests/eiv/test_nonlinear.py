import numpy as np

from wphid.eiv.nonlinear import fit_nonlinear_simex_path


def test_nonlinear_simex_rebuilds_every_level_and_returns_bounded_path():
    temperature = np.linspace(600.0, 2100.0, 120)
    target = 0.2 + 4.0*np.exp(-7000.0/temperature)
    calls = []
    def reconstruction(level, replicate):
        calls.append((level, replicate))
        rng = np.random.default_rng(1000+replicate+int(level*10))
        observed = temperature+rng.normal(0.0, np.sqrt(level)*2.0, temperature.size)
        return (np.ones((temperature.size, 1)), target,
                lambda parameter: np.exp(-parameter[0]/observed))
    result = fit_nonlinear_simex_path(
        [0.0, 0.5, 1.0, 1.5], reconstruction, [(3000.0, 12000.0)],
        n_replicates=3, seed=9,
    )
    assert len(calls) == 12
    assert result.parameter_path.shape == (4, 1)
    assert 3000.0 <= result.extrapolated_parameters[0] <= 12000.0


def test_nonlinear_simex_rejects_invalid_levels():
    try:
        fit_nonlinear_simex_path([0.0, 1.0], lambda *_: None, [(0.0, 1.0)],
                                 n_replicates=1, seed=1)
    except ValueError:
        return
    raise AssertionError("invalid levels must raise ValueError")


def test_weight_concentration_is_diagnostic_not_a_geometry_failure():
    levels = [0.0, 0.5, 1.0, 1.5]
    calls = []

    def reconstruction(level, replicate):
        calls.append((level, replicate))
        x = np.linspace(0.2, 1.0, 40)
        target = 1.0 + 2.0*np.exp(-0.7*x)
        return np.ones((x.size, 1)), target, lambda p: np.exp(-p[0]*x)

    result = fit_nonlinear_simex_path(
        levels, reconstruction, [(0.1, 2.0)], n_replicates=4, seed=4,
        variance_weighted=True, path_order_candidates=(1, 2),
        bootstrap_replicates=10,
    )
    assert max(result.diagnostics["path_geometry_condition_numbers"]) < 1.0e4
    assert "weighted_design_condition_numbers" in result.diagnostics


def _deterministic_linear_reconstruction(level, replicate):
    parameter = 1.0 + 0.1*level
    replicate_offsets = (-0.015, -0.005, 0.005, 0.015)
    parameter += replicate_offsets[replicate]
    coordinate = np.linspace(0.1, 2.0, 80)
    base = np.ones((coordinate.size, 1))
    target = (0.25 + 2.0*np.exp(-parameter*coordinate)
              + 1e-5*np.sin(7.0*coordinate))
    return base, target, lambda candidate: np.exp(-candidate[0]*coordinate)


def test_weighted_path_selection_and_extrapolation_are_deterministic():
    result = fit_nonlinear_simex_path(
        [0.0, 0.5, 1.0, 1.5],
        _deterministic_linear_reconstruction,
        [(0.2, 2.0)],
        n_replicates=4,
        seed=71,
        variance_weighted=True,
        path_order_candidates=(1, 2),
        maximum_relative_extrapolation=1.0,
    )
    assert result.selected_path_orders.tolist() == [2]
    assert np.isclose(result.extrapolated_parameters[0], 0.9, atol=2e-3)
    assert result.diagnostics["variance_weighted"] is True
    assert all(values.shape == (4, 1) for values in result.replicate_parameters)


def test_bootstrap_diagnostics_are_finite_and_reproducible():
    keyword_arguments = dict(
        levels=[0.0, 0.5, 1.0, 1.5],
        reconstruction=_deterministic_linear_reconstruction,
        bounds=[(0.2, 2.0)],
        n_replicates=4,
        seed=19,
        variance_weighted=True,
        path_order_candidates=(1,),
        bootstrap_replicates=40,
        maximum_relative_extrapolation=1.0,
    )
    first = fit_nonlinear_simex_path(**keyword_arguments)
    second = fit_nonlinear_simex_path(**keyword_arguments)
    assert first.bootstrap_extrapolated_parameters.shape == (40, 1)
    assert first.extrapolated_covariance.shape == (1, 1)
    assert first.extrapolated_standard_errors.shape == (1,)
    assert np.all(np.isfinite(first.extrapolated_covariance))
    assert np.all(first.extrapolated_standard_errors > 0.0)
    assert first.extrapolated_coefficients.shape == (2,)
    assert first.coefficient_path.shape == (4, 2)
    assert first.bootstrap_extrapolated_coefficients.shape == (40, 2)
    assert first.coefficient_covariance.shape == (2, 2)
    assert np.all(np.isfinite(first.coefficient_covariance))
    np.testing.assert_allclose(
        first.bootstrap_extrapolated_parameters,
        second.bootstrap_extrapolated_parameters,
    )
    np.testing.assert_allclose(
        first.bootstrap_extrapolated_coefficients,
        second.bootstrap_extrapolated_coefficients,
    )


def test_historical_default_call_remains_compatible():
    result = fit_nonlinear_simex_path(
        [0.0, 0.5, 1.0, 1.5],
        _deterministic_linear_reconstruction,
        [(0.2, 2.0)],
        n_replicates=4,
        seed=5,
        maximum_relative_extrapolation=1.0,
    )
    assert result.selected_path_orders.tolist() == [2]
    assert result.extrapolated_standard_errors is None
    assert result.extrapolated_covariance is None
    assert result.bootstrap_extrapolated_parameters is None
