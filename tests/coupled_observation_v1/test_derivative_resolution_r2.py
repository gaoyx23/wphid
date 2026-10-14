"""C1 zero-flux cusp and complete-map reference, with unchanged tolerances."""

import importlib.util
from pathlib import Path

import numpy as np

from wphid.coupled_balance_v1.noise import numerical_balance_linearization
from wphid.coupled_balance_v1.operators import attenuated_flux
from wphid.coupled_observation_v1 import linearize_balances


spec = importlib.util.spec_from_file_location("frozen_observation_fixture_r2", Path(__file__).with_name("test_observation.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_zero_flux_reference_cancels_the_cusp_first_order_difference_error():
    b, gain, h = .005, 5., .004
    value = np.array([0.])
    d1 = (attenuated_flux(value+gain*h, b)[0]-attenuated_flux(value-gain*h, b)[0])/(2*h)
    d2 = (attenuated_flux(value+gain*h/2, b)[0]-attenuated_flux(value-gain*h/2, b)[0])/h
    assert abs(d1[0]-gain) > 1e-4
    # Q(v)=v-b*v*abs(v)+2*b^2*v^3+...; 2D(h/2)-D(h)
    # cancels the C1 cusp O(h) term. Fixed bound, no step selection by results.
    error_bound = 2*b*b*(gain*h)**2*gain
    np.testing.assert_allclose(2*d2-d1, [gain], rtol=0., atol=error_bound)


def test_complete_column_derivatives_with_cusp_aware_fixed_two_scale_reference():
    p, layout, raw, inputs, mapper = fixture.problem()
    states, args = inputs(raw)
    baseline, analytic, _ = linearize_balances(p, layout, states, **args)
    steps = 2e-5*np.maximum(np.abs(raw), 1.)
    for key in ("pO2", "pH2O", "pProducts", "external_o", "external_w"):
        steps[layout.slices[key]] = np.minimum(steps[layout.slices[key]], .004)
    reference, coarse = numerical_balance_linearization(mapper, raw, layout.coordinate_keys, steps)
    _, fine = numerical_balance_linearization(mapper, raw, layout.coordinate_keys, steps/2)
    np.testing.assert_array_equal(baseline.target, reference.target)
    np.testing.assert_array_equal(baseline.matrix, reference.matrix)
    for actual, large, small in zip(analytic.column_jacobians, coarse.column_jacobians, fine.column_jacobians):
        expected = (2*small-large).toarray()
        numerical_scale = max(float(np.max(np.abs(expected))), 1e-20)
        np.testing.assert_allclose(actual.toarray()/numerical_scale, expected/numerical_scale, rtol=2e-6, atol=2e-8)
