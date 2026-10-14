"""Shared coordinates, correlated windows and full balance observation checks."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from wphid.coupled_balance_v1 import (
    BoundaryInputs, Modulation, ShellMesh, TransportShape, WeakProjector,
    WeakSystem, arrhenius_pair, assemble_balances, temporal_tests,
)
from wphid.coupled_balance_v1.assembler import COLUMNS, FIELDS, PRESSURES
from wphid.coupled_balance_v1.noise import (
    CovarianceRankFailure, SharedObservationLinearization,
    numerical_balance_linearization, observation_factor,
)
from wphid.coupled_balance_v1.operators import SIDES


def test_same_target_and_design_observation_can_cancel_noise_exactly():
    linearization = SharedObservationLinearization(np.eye(2), (np.eye(2),), ("a", "b"))
    noise = linearization.noise_operator([1.], observation_factor([1., 1.]))
    np.testing.assert_array_equal(noise.dense(), np.zeros((2, 2)))
    with pytest.raises(CovarianceRankFailure):
        noise.whitener()


def test_overlapping_weak_rows_retain_off_diagonal_covariance():
    jacobian = np.array([[1., 1., 0.], [0., 1., 1.]])
    linearization = SharedObservationLinearization(jacobian, (), ("y0", "y1", "y2"))
    noise = linearization.noise_operator([], observation_factor([1., 4., 9.]))
    np.testing.assert_allclose(noise.dense(), [[5., 4.], [4., 13.]])
    np.testing.assert_allclose(noise.matvec([2., -1.]), [6., -5.])


def test_common_channel_error_factor_is_not_dropped():
    jacobian = np.array([[1., 1., 0.], [0., 1., 1.]])
    shared = np.array([[.1], [.2], [.3]])
    variance = np.array([.01, .02, .03])
    linearization = SharedObservationLinearization(jacobian, (), ("carbon", "oxygen", "temperature"))
    noise = linearization.noise_operator([], observation_factor(variance, shared))
    expected = jacobian @ (np.diag(variance)+shared @ shared.T) @ jacobian.T
    np.testing.assert_allclose(noise.dense(), expected)
    np.testing.assert_allclose(noise.diagonal(), np.diag(expected))


def test_residual_covariance_depends_on_the_current_linear_coefficients():
    design = np.array([[0., 1.], [1., 0.]])
    linearization = SharedObservationLinearization(np.eye(2), (design,), ("a", "b"))
    for amplitude in (.2, .7):
        covariance = linearization.noise_operator([amplitude], observation_factor([1., 1.])).dense()
        np.testing.assert_allclose(covariance, [[1+amplitude**2, -2*amplitude], [-2*amplitude, 1+amplitude**2]])


def test_positive_definite_whitening_preserves_the_declared_covariance():
    linearization = SharedObservationLinearization([[1., 1., 0.], [0., 1., 1.]], (), ("a", "b", "c"))
    noise = linearization.noise_operator([], observation_factor([1., 4., 9.]))
    whitener = noise.whitener()
    np.testing.assert_allclose(whitener @ noise.dense() @ whitener.T, np.eye(2), atol=1e-12)


def test_duplicate_observation_ids_cannot_create_independent_copies():
    with pytest.raises(ValueError, match="unique"):
        SharedObservationLinearization(np.eye(2), (), ("same", "same"))


def test_sparse_covariance_actions_do_not_require_a_dense_global_matrix():
    linearization = SharedObservationLinearization(csr_matrix(np.eye(3)), (), ("a", "b", "c"))
    noise = linearization.noise_operator([], observation_factor([1., 2., 3.]))
    np.testing.assert_array_equal(noise.matvec([1., 1., 1.]), [1., 2., 3.])
    with pytest.raises(ValueError, match="cap"):
        noise.dense(max_rows=2)


def full_mapper(observation):
    nt, ns, nx = 5, 2, 1
    states = dict(zip(FIELDS, observation.reshape(5, nt, ns, nx)))
    mesh = ShellMesh(np.ones((ns, nx)), np.ones((ns-1, nx)), np.empty((ns, nx-1)),
                     np.ones((ns-1, nx)), np.empty((ns, nx-1)),
                     {"inner": np.ones(ns), "outer": np.ones(ns), "inlet": np.ones(nx), "outlet": np.ones(nx)})
    projector = WeakProjector(mesh, temporal_tests(np.arange(nt, dtype=float), window_samples=5, stride_samples=1),
                              np.array([[[1.], [-.3]]]))
    rates = arrhenius_pair(states, oxygen_energy=1.1e5, steam_energy=1.3e5,
                           oxygen_order=1.1, steam_order=.8, carbon_order=1.2)
    loss = 1-states["carbon"]
    gas = Modulation(1+.7*loss)
    thermal = Modulation(1-.2*loss)
    external = {("pO2", "inner"): np.full((nt, ns), 3e4),
                ("pH2O", "inner"): np.full((nt, ns), 7e5),
                ("pProducts", "inner"): np.zeros((nt, ns)),
                ("pO2", "outer"): np.full((nt, ns), 2.2e4)}
    barriers = {key: Modulation(np.ones((nt, ns))) for key in external}
    sealed = frozenset((k, s) for k in PRESSURES for s in SIDES if (k, s) not in external)
    boundary = BoundaryInputs(external, barriers, sealed, {"inner": np.full((nt, ns), 60.)},
                              {"outer": np.full((nt, ns), 300.)}, frozenset(("inlet", "outlet")))
    return assemble_balances(projector, states, rates, relative_rate=1.4,
        transport=TransportShape(1.3, .8, 1.2, .005), gas_modulation=gas,
        thermal_modulation=thermal, thermal_direction_ratio=1.7, heat_knots=(400., 1000.), boundary=boundary)


def test_complete_observation_map_includes_storage_sources_transport_and_radiation():
    tt, ii, _ = np.meshgrid(np.arange(5.), np.arange(2.), np.arange(1.), indexing="ij")
    values = np.stack((.9-.001*tt+.002*ii, 2e4+200*tt+350*ii,
                       6e5+800*tt+1200*ii, 500+10*tt+5*ii, 500+10*tt+3*ii)).ravel()
    steps = np.repeat([1e-6, 1e-3, 1e-2, 1e-3, 1e-3], 10)
    keys = tuple(f"{field}:{index}" for field in FIELDS for index in range(10))
    baseline, linearization = numerical_balance_linearization(full_mapper, values, keys, steps)
    coefficients = np.array([.1, 2e4, 3e5, 4e4, 10., 1e-6, 1e-3, 1e-6, 1e-5, 1e-12, .5])
    direction = .2*steps*np.sin(np.arange(len(values))+.4)
    predicted = linearization.residual_jacobian(coefficients) @ direction
    measured = (full_mapper(values+direction).residual(coefficients)-
                full_mapper(values-direction).residual(coefficients))/2
    np.testing.assert_allclose(predicted, measured, rtol=2e-5, atol=2e-8)
    assert baseline.column_keys == COLUMNS
    covariance = linearization.noise_operator(coefficients, observation_factor(steps**2)).dense()
    assert np.max(np.abs(covariance-np.diag(np.diag(covariance)))) > 0


def test_nonsmooth_physical_boundary_failure_is_not_silently_clipped():
    def mapper(value):
        if value[0] < 0:
            raise ValueError("outside physical domain")
        return WeakSystem(np.zeros(1), np.array([[np.sqrt(value[0])]]), {}, (("carbon", 0, 0),), ("source",))
    with pytest.raises(ValueError, match="physical domain"):
        numerical_balance_linearization(mapper, [0.], ("pressure",), [1e-6])
