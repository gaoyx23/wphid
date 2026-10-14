"""One independent derivative-resolution repair; original 19 passes not rerun."""

import importlib.util
from pathlib import Path

import numpy as np

from wphid.coupled_balance_v1.assembler import COLUMNS
from wphid.coupled_balance_v1.noise import numerical_balance_linearization
from wphid.coupled_observation_v1 import linearize_balances


spec = importlib.util.spec_from_file_location("frozen_observation_fixture", Path(__file__).with_name("test_observation.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_complete_analytic_derivatives_with_pressure_steps_bounded_by_flux_curvature():
    p, layout, raw, inputs, mapper = fixture.problem()
    states, args = inputs(raw)
    baseline, analytic, _ = linearize_balances(p, layout, states, **args)
    steps = 2e-5*np.maximum(np.abs(raw), 1.)
    # For Q, Q'''/Q' = 12 b^2/(1+4b|v|)^2 away from zero. With
    # b=.005, the fixture face gains <5 and h<=.01 Pa, the leading
    # centered relative derivative defect is bounded by 1.25e-7.
    # The original h=12 Pa used absolute steam level, not face differences.
    # Keep the original derivative tolerances; change no scientific threshold.
    for key in ("pO2", "pH2O", "pProducts", "external_o", "external_w"):
        steps[layout.slices[key]] = np.minimum(steps[layout.slices[key]], .01)
    reference, numerical = numerical_balance_linearization(mapper, raw, layout.coordinate_keys, steps)
    np.testing.assert_array_equal(baseline.target, reference.target)
    np.testing.assert_array_equal(baseline.matrix, reference.matrix)
    np.testing.assert_allclose(analytic.target_jacobian.toarray(), numerical.target_jacobian.toarray(), atol=2e-10)
    for actual, expected in zip(analytic.column_jacobians, numerical.column_jacobians):
        expected = expected.toarray()
        numerical_scale = max(float(np.max(np.abs(expected))), 1e-20)
        np.testing.assert_allclose(actual.toarray()/numerical_scale, expected/numerical_scale, rtol=2e-6, atol=2e-8)


def test_internal_transport_observation_derivative_annihilates_constant_pressure_translation():
    p, layout, raw, inputs, _ = fixture.problem()
    states, args = inputs(raw)
    _, derivative, _ = linearize_balances(p, layout, states, **args)
    direction = np.zeros(layout.size)
    direction[layout.slices["pH2O"]] = 1.
    actual = derivative.column_jacobians[COLUMNS.index("gas_scale")] @ direction
    np.testing.assert_allclose(actual, 0., atol=1e-12)
