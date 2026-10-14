"""New analytic-map qualifications; no old tests or scientific cases repeated."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye, kron

from wphid.coupled_balance_v1 import (
    Modulation, ShellMesh, TransportShape, WeakProjector,
    arrhenius_pair, assemble_balances, temporal_tests,
)
from wphid.coupled_balance_v1.assembler import COLUMNS, FIELDS, PRESSURES
from wphid.coupled_balance_v1.noise import numerical_balance_linearization
from wphid.coupled_balance_v1.operators import SIDES
from wphid.coupled_observation_v1 import (
    KineticPoint, ObservationLayout, ObservedBoundary, ObservedValue,
    SourceDerivativeFailure, SparseWeakMaps, linearize_balances,
)


def projector(ns=2, nx=2):
    mesh = ShellMesh(np.full((ns, nx), .7), np.full((ns-1, nx), 1.2),
        np.full((ns, nx-1), .8), np.full((ns-1, nx), .4), np.full((ns, nx-1), .3),
        {"inner": np.full(ns, .5), "outer": np.full(ns, .6),
         "inlet": np.full(nx, .8), "outlet": np.full(nx, .9)})
    spatial = np.eye(ns*nx).reshape(ns*nx, ns, nx)
    return WeakProjector(mesh, temporal_tests(np.arange(5.), window_samples=3, stride_samples=1), spatial)


@pytest.mark.parametrize("ns,nx", [(2, 2), (1, 2), (2, 1), (1, 1)])
def test_sparse_storage_and_sources_equal_declared_contractions(ns, nx):
    p = projector(ns, nx)
    maps = SparseWeakMaps(p)
    values = (1+np.arange(np.prod(p.state_shape))).reshape(p.state_shape)/17
    np.testing.assert_allclose(maps.storage @ values.ravel(), p.storage(values), atol=1e-14)
    np.testing.assert_allclose(maps.source @ values.ravel(), p.source(values), atol=1e-14)


def test_sparse_face_maps_equal_gradient_average_and_conservative_flux():
    p = projector()
    maps = SparseWeakMaps(p)
    values = np.cos(np.arange(20.)/3).reshape(p.state_shape)
    gradients, averages = p.gradient(values), p.face_modulation(values)
    for index, axis in enumerate(("s", "x")):
        np.testing.assert_allclose(maps.gradient[axis] @ values.ravel(), gradients[index].ravel())
        np.testing.assert_allclose(maps.average[axis] @ values.ravel(), averages[index].ravel())
    actual = maps.flux["s"] @ gradients[0].ravel()+maps.flux["x"] @ gradients[1].ravel()
    np.testing.assert_allclose(actual, p.flux(*gradients), atol=1e-14)


@pytest.mark.parametrize("side", SIDES)
def test_sparse_boundary_maps_preserve_each_declared_trace(side):
    p = projector()
    maps = SparseWeakMaps(p)
    values = np.arange(20.).reshape(p.state_shape)
    trace = p.trace(values, side)
    np.testing.assert_array_equal(maps.trace[side] @ values.ravel(), trace.ravel())
    np.testing.assert_allclose(maps.boundary[side] @ trace.ravel(), p.boundary(trace, side))


def problem():
    p = projector()
    blocks = [(key, p.state_shape) for key in FIELDS]
    blocks += [(key, (5, 2)) for key in ("external_o", "external_w", "heat", "ambient")]
    blocks += [("damage", p.state_shape)]
    layout = ObservationLayout(blocks)
    tt, ss, xx = np.indices(p.state_shape)
    arrays = {"carbon": .9-.005*tt+.002*ss, "pO2": 2e4+100*tt+200*ss+70*xx,
              "pH2O": 6e5+500*tt+300*ss+50*xx, "pProducts": 1e3+20*tt+30*ss,
              "T": 650+10*tt+20*ss+15*xx, "external_o": np.full((5, 2), 2.8e4),
              "external_w": np.full((5, 2), 7e5), "heat": np.full((5, 2), 100.),
              "ambient": np.full((5, 2), 300.), "damage": .1+.002*tt+.003*xx}
    raw = np.concatenate([arrays[key].ravel() for key in layout.blocks])
    point = KineticPoint(1.1e5, 1.3e5, 1.1, .8, 1.2)
    maps = SparseWeakMaps(p)

    def inputs(observation):
        observed = {key: layout.observed(key, observation[layout.slices[key]].reshape(shape))
                    for key, shape in layout.blocks.items()}
        states = {key: observed[key] for key in FIELDS}
        # A declared linear reconstruction is included in both target and design.
        reconstruction = kron(eye(5, format="csr"), csr_matrix(
            [[.8, .2, 0., 0.], [.2, .8, 0., 0.], [0., 0., .8, .2], [0., 0., .2, .8]]), format="csr")
        carbon = states["carbon"]
        states["carbon"] = ObservedValue((reconstruction @ carbon.value.ravel()).reshape(p.state_shape),
                                        reconstruction @ carbon.jacobian)
        carbon, damage = states["carbon"], observed["damage"]
        gas = ObservedValue(1+.6*(1-carbon.value)+.2*damage.value, -.6*carbon.jacobian+.2*damage.jacobian)
        thermal = ObservedValue(1-.15*(1-carbon.value)+.05*damage.value, .15*carbon.jacobian+.05*damage.jacobian)
        external = {("pO2", "inner"): observed["external_o"],
                    ("pO2", "outer"): ObservedValue(1.1*observed["external_o"].value, 1.1*observed["external_o"].jacobian),
                    ("pH2O", "inner"): observed["external_w"],
                    ("pProducts", "inner"): ObservedValue.fixed(np.zeros((5, 2)), layout.size)}
        barriers = {}
        for key, side in external:
            trace = maps.trace[side]
            barrier_value = (1+.25*(1-trace @ carbon.value.ravel())+.3*(trace @ damage.value.ravel())).reshape(5, 2)
            barriers[(key, side)] = ObservedValue(barrier_value, trace @ (-.25*carbon.jacobian+.3*damage.jacobian))
        sealed = frozenset((key, side) for key in PRESSURES for side in SIDES if (key, side) not in external)
        boundary = ObservedBoundary(external, barriers, sealed, {"inner": observed["heat"]},
                                    {"outer": observed["ambient"]}, frozenset(("inlet", "outlet")))
        return states, dict(kinetics=point, relative_rate=1.4, transport=TransportShape(1.3, .8, 1.2, .005),
            gas_modulation=gas, thermal_modulation=thermal, thermal_direction_ratio=1.7,
            heat_knots=(400., 1000.), boundary=boundary, maps=maps)

    def mapper(observation):
        states, args = inputs(observation)
        values = {key: item.value for key, item in states.items()}
        rates = arrhenius_pair(values, **point.arguments())
        return assemble_balances(p, values, rates, relative_rate=args["relative_rate"],
            transport=args["transport"], gas_modulation=Modulation(args["gas_modulation"].value),
            thermal_modulation=Modulation(args["thermal_modulation"].value),
            thermal_direction_ratio=args["thermal_direction_ratio"], heat_knots=args["heat_knots"],
            boundary=args["boundary"].values())
    return p, layout, raw, inputs, mapper


def test_all_eleven_analytic_column_maps_match_independently_recomputed_observations():
    p, layout, raw, inputs, mapper = problem()
    states, args = inputs(raw)
    baseline, analytic, _ = linearize_balances(p, layout, states, **args)
    reference, numerical = numerical_balance_linearization(mapper, raw, layout.coordinate_keys,
        2e-5*np.maximum(np.abs(raw), 1.))
    np.testing.assert_array_equal(baseline.target, reference.target)
    np.testing.assert_array_equal(baseline.matrix, reference.matrix)
    np.testing.assert_allclose(analytic.target_jacobian.toarray(), numerical.target_jacobian.toarray(), atol=2e-10)
    for actual, expected in zip(analytic.column_jacobians, numerical.column_jacobians):
        expected = expected.toarray()
        numerical_scale = max(float(np.max(np.abs(expected))), 1e-20)
        np.testing.assert_allclose(actual.toarray()/numerical_scale, expected/numerical_scale, rtol=2e-6, atol=2e-8)


def zero_problem(order, *, active=True, fixed=False):
    p, layout, raw, inputs, _ = problem()
    states, args = inputs(raw)
    value = states["pH2O"].value.copy()
    value[2 if active else 0, 0, 0] = 0.
    jacobian = states["pH2O"].jacobian.copy().tolil()
    if fixed:
        jacobian[2*4, :] = 0.
    states["pH2O"] = ObservedValue(value, jacobian.tocsr())
    args["kinetics"] = KineticPoint(1.1e5, 1.3e5, 1.1, order, 1.2)
    return p, layout, states, args


def test_fractional_zero_at_active_variable_source_is_retained_as_failure():
    p, layout, states, args = zero_problem(.8)
    with pytest.raises(SourceDerivativeFailure) as error:
        linearize_balances(p, layout, states, **args)
    assert error.value.report["unbounded_active_derivative_nodes"] == 1


def test_fractional_initial_zero_keeps_storage_coordinate_and_records_unqualified_law():
    p, layout, states, args = zero_problem(.8, active=False)
    _, derivative, report = linearize_balances(p, layout, states, **args)
    domain = report["source_pressure_domains"]["pH2O"]
    assert domain["zero_nodes_outside_source_support"] == 1
    assert domain["zero_nodes_used_by_storage"] == 1
    index = layout.slices["pH2O"].start
    assert derivative.target_jacobian[:, index].nnz > 0
    assert not domain["raw_zero_observation_law_qualified"]
    assert report["zero_nodes_removed_from_observations"] == 0


def test_fixed_fractional_zero_does_not_create_a_random_pressure_direction():
    p, layout, states, args = zero_problem(.8, fixed=True)
    _, _, report = linearize_balances(p, layout, states, **args)
    assert report["source_pressure_domains"]["pH2O"]["variable_zero_nodes"] == 0


@pytest.mark.parametrize("order", [1., 1.7])
def test_integer_or_superlinear_pressure_zero_has_declared_finite_tangent(order):
    p, layout, states, args = zero_problem(order)
    _, derivative, report = linearize_balances(p, layout, states, **args)
    node = 2*4
    index = layout.slices["pH2O"].start+node
    column = derivative.column_jacobians[COLUMNS.index("steam_response")][:, index].toarray().ravel()
    expected = np.zeros_like(column)
    if order == 1:
        slope = np.exp(-1.3e5/(8.314462618*states["T"].value.ravel()[node]))*states["carbon"].value.ravel()[node]**1.2/1e6
        block = int(np.prod(p.row_shape))
        expected[2*block:3*block] = -args["maps"].source[:, node].toarray().ravel()*slope
    np.testing.assert_allclose(column, expected, atol=1e-24)
    assert not report["source_pressure_domains"]["pH2O"]["raw_zero_observation_law_qualified"]


def test_shared_external_load_is_one_raw_block_for_both_boundaries():
    p, layout, raw, inputs, _ = problem()
    states, args = inputs(raw)
    _, derivative, _ = linearize_balances(p, layout, states, **args)
    block = layout.slices["external_o"]
    actual = derivative.column_jacobians[COLUMNS.index("surface_transfer")][:, block]
    assert actual.nnz > 0
    assert len(set(derivative.coordinate_keys)) == layout.size
    assert sum(key.startswith("external_o[") for key in derivative.coordinate_keys) == 10


def test_auxiliary_observed_damage_cannot_be_omitted_from_transport_noise():
    p, layout, raw, inputs, _ = problem()
    states, args = inputs(raw)
    _, derivative, _ = linearize_balances(p, layout, states, **args)
    block = layout.slices["damage"]
    for column in ("gas_scale", "thermal_scale", "surface_transfer"):
        assert derivative.column_jacobians[COLUMNS.index(column)][:, block].nnz > 0


def test_observation_width_mismatch_is_rejected():
    p, layout, raw, inputs, _ = problem()
    states, args = inputs(raw)
    args["gas_modulation"] = ObservedValue.fixed(args["gas_modulation"].value, layout.size-1)
    with pytest.raises(ValueError, match="same raw"):
        linearize_balances(p, layout, states, **args)


def test_another_projectors_maps_cannot_be_silently_reused():
    p, layout, raw, inputs, _ = problem()
    states, args = inputs(raw)
    args["maps"] = SparseWeakMaps(projector())
    with pytest.raises(ValueError, match="exact projector"):
        linearize_balances(p, layout, states, **args)


def test_duplicate_raw_blocks_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        ObservationLayout([("a", (2,)), ("a", (2,))])
