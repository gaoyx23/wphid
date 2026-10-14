"""Physical syntax, active source graphs and their new derivatives only."""

from types import SimpleNamespace

import numpy as np
import pytest

from wphid.coupled_balance_v1 import ShellMesh, TransportShape, WeakProjector, temporal_tests
from wphid.coupled_balance_v1.assembler import FIELDS, PRESSURES
from wphid.coupled_balance_v1.noise import numerical_balance_linearization
from wphid.coupled_balance_v1.operators import SIDES
from wphid.coupled_observation_v1 import ObservationLayout, ObservedBoundary, ObservedValue, SparseWeakMaps, SourceDerivativeFailure
from wphid.coupled_candidates_v1 import (
    ModulationKink, RateSpec, SOURCE_CANDIDATES, assemble_candidate,
    exponential_barrier, gas_modulation, thermal_modulation,
)
from wphid.transport_mechanisms import modulation_fraction


def problem():
    nt, ns, nx = 5, 2, 1
    shape = (nt, ns, nx)
    mesh = ShellMesh(np.full((ns, nx), .7), np.full((ns-1, nx), .8), np.empty((ns, nx-1)),
        np.full((ns-1, nx), .4), np.empty((ns, nx-1)),
        {"inner": np.full(ns, .5), "outer": np.full(ns, .6), "inlet": np.full(nx, .7), "outlet": np.full(nx, .8)})
    projector = WeakProjector(mesh, temporal_tests(np.arange(nt, dtype=float), window_samples=3, stride_samples=1),
                              np.eye(ns*nx).reshape(ns*nx, ns, nx))
    layout = ObservationLayout([(key, shape) for key in (*FIELDS, "crack", "stress")])
    tt, ss, xx = np.indices(shape)
    arrays = {"T": 650.+13*tt+27*ss+9*xx, "carbon": .93-.002*tt-.001*ss,
              "pO2": 2e4+200*tt+300*ss, "pH2O": 6e5+300*tt+100*ss,
              "pProducts": 1e3+40*tt+20*ss, "crack": .15+.003*tt+.02*ss,
              "stress": 2e5+2e4*tt+1e4*ss+500*xx}
    raw = np.concatenate([arrays[key].ravel() for key in layout.blocks])
    gas_point = dict(stress_gain_per_pa=2e-6, stress_cap=2., damage_gain=3., carbon_gain=2.,
                     closure_gain=.2, closure_temperature=680., closure_span=160., total_cap=4.)
    thermal_point = dict(loss=.35, stiffness_floor=.15, conductivity_floor=.3)
    maps = SparseWeakMaps(projector)

    def build(observation, candidate="both_shared", ratio=2.3, overrides=None):
        overrides = {} if overrides is None else overrides
        drivers = {key: layout.observed(key, observation[layout.slices[key]].reshape(shape)) for key in layout.blocks}
        gp = {key: overrides.get("gas.modulation."+key, value) for key, value in gas_point.items()}
        hp = {key: overrides.get("thermal.modulation."+key, value) for key, value in thermal_point.items()}
        gas, thermal = gas_modulation("combined_opening", drivers, gp), thermal_modulation("stiffness", drivers, hp)
        boundary_shape = (nt, ns)
        external = {("pO2", "inner"): ObservedValue.fixed(np.full(boundary_shape, 2.8e4), layout.size),
                    ("pH2O", "inner"): ObservedValue.fixed(np.full(boundary_shape, 7e5), layout.size),
                    ("pProducts", "inner"): ObservedValue.fixed(np.zeros(boundary_shape), layout.size),
                    ("pO2", "outer"): ObservedValue.fixed(np.full(boundary_shape, 2.1e4), layout.size)}
        coverage = ObservedValue((maps.trace["inner"] @ drivers["crack"].value.ravel()).reshape(boundary_shape),
                                  maps.trace["inner"] @ drivers["crack"].jacobian)
        barrier = exponential_barrier(coverage, overrides.get("boundary.barrier_strength", .2))
        barriers = {key: (barrier.observed_value() if key[1] == "inner" else
                         ObservedValue.fixed(np.ones(boundary_shape), layout.size)) for key in external}
        barrier_parameters = {key: barrier.parameters for key in external if key[1] == "inner"}
        sealed = frozenset((field, side) for field in PRESSURES for side in SIDES if (field, side) not in external)
        boundary = ObservedBoundary(external, barriers, sealed,
            {"inner": ObservedValue.fixed(np.full(boundary_shape, 100.), layout.size)},
            {"outer": ObservedValue.fixed(np.full(boundary_shape, 300.), layout.size)}, frozenset(("inlet", "outlet")))
        specs = {}
        for path, energy, order in (("oxygen", 1.1e5, 1.1), ("steam", 1.3e5, .8)):
            if path in SOURCE_CANDIDATES[candidate]:
                specs[path] = RateSpec(1e5*overrides.get("kinetic."+path+".energy_1e5", energy/1e5),
                    overrides.get("kinetic."+path+".pressure_order", order), overrides.get("kinetic.carbon_order", 1.2))
        use_ratio = len(specs) == 2 and candidate != "both_free"
        return assemble_candidate(projector, layout, {key: drivers[key] for key in FIELDS},
            source_candidate=candidate, rate_specs=specs,
            relative_rate=overrides.get("reaction.relative_rate", ratio) if use_ratio else None,
            shared_carbon_order=True, transport=TransportShape(1.3, .8, 1.2, 0.), flux_law="Darcy",
            gas=gas, thermal=thermal, thermal_direction_ratio=1.7, heat_knots=(400., 1000.),
            boundary=boundary, barrier_parameters=barrier_parameters, maps=maps)
    return projector, layout, raw, gas_point, thermal_point, build


@pytest.mark.parametrize("candidate,count", [("null", 6), ("oxygen_only", 10), ("steam_only", 10),
    ("both_shared", 11), ("both_free", 14), ("both_swapped_product", 11), ("both_oxygen_heat", 11)])
def test_each_active_graph_has_only_its_declared_columns_and_kinetic_coordinates(candidate, count):
    _, _, raw, _, _, build = problem()
    model, linearization, report = build(raw, candidate)
    assert model.matrix.shape[1] == count
    assert len(linearization.column_jacobians) == count
    assert report["active_paths"] == list(SOURCE_CANDIDATES[candidate])
    for path in ("oxygen", "steam"):
        if path not in SOURCE_CANDIDATES[candidate]:
            assert not any(key.startswith("kinetic."+path+".") for key in model.matrix_jacobian)
    assert ("reaction.relative_rate" in model.matrix_jacobian) == (count == 11)
    assert "gas.forchheimer_product" not in model.matrix_jacobian


def test_source_graphs_share_rows_and_targets_for_fair_candidate_comparison():
    _, _, raw, _, _, build = problem()
    reference = build(raw)[0]
    for candidate in SOURCE_CANDIDATES:
        model = build(raw, candidate)[0]
        assert model.row_keys == reference.row_keys
        np.testing.assert_array_equal(model.target, reference.target)


def test_free_responses_can_embed_shared_mixture_without_being_forced_to_it():
    _, _, raw, _, _, build = problem()
    shared, free = build(raw)[0], build(raw, "both_free")[0]
    a, ratio, product, heat = 2., 2.3, 4., 5.
    shared_coeff = {"carbon_rate_scale": a, "oxygen_response": 3., "steam_response": 6.,
                    "product_response": product, "heat_response": heat}
    free_coeff = {"carbon.oxygen_rate_scale": a, "carbon.steam_rate_scale": a*ratio,
                  "oxygen_response": 3., "steam_response": 6.,
                  "product.oxygen_response": product, "product.steam_response": product*ratio,
                  "heat.oxygen_response": heat, "heat.steam_response": heat*ratio}
    left = shared.matrix @ np.array([shared_coeff.get(key, .2) for key in shared.column_keys])
    right = free.matrix @ np.array([free_coeff.get(key, .2) for key in free.column_keys])
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-8)


def test_wrong_product_graph_differs_and_has_a_declared_equivalence_at_unit_ratio():
    _, _, raw, _, _, build = problem()
    shared, swapped = build(raw)[0], build(raw, "both_swapped_product")[0]
    index = shared.column_keys.index("product_response")
    assert np.linalg.norm(shared.matrix[:, index]-swapped.matrix[:, index]) > 0
    np.testing.assert_allclose(build(raw, ratio=1.)[0].matrix,
                               build(raw, "both_swapped_product", ratio=1.)[0].matrix)


def test_inactive_fractional_path_is_not_evaluated_but_pressure_storage_is_retained():
    _, layout, raw, _, _, build = problem()
    raw[layout.slices["pH2O"].start+4] = 0.
    model, derivative, _ = build(raw, "oxygen_only")
    assert derivative.target_jacobian[:, layout.slices["pH2O"].start+4].nnz > 0
    assert "steam_response" not in model.column_keys
    with pytest.raises(SourceDerivativeFailure):
        build(raw, "both_shared")


def test_combined_product_coordinates_equal_archived_opening_formula():
    _, layout, raw, _, _, _ = problem()
    fields = {key: layout.observed(key, raw[selection].reshape(layout.blocks[key])) for key, selection in layout.slices.items()}
    ratio, stress_gain, threshold = 7., 1.3, 3e5
    transport = SimpleNamespace(open_to_closed_diffusivity_ratio=ratio, stress_opening_gain=stress_gain,
        damage_opening_gain=3.2, carbon_loss_opening_gain=2.1, high_temperature_closure=.4,
        closure_temperature=680., closure_temperature_span=160.)
    point = dict(stress_gain_per_pa=(ratio-1)*stress_gain/(9.6*threshold),
        stress_cap=(ratio-1)*stress_gain*3/9.6, damage_gain=(ratio-1)*3.2/9.6,
        carbon_gain=(ratio-1)*2.1/9.6, closure_gain=(ratio-1)*.4/9.6,
        closure_temperature=680., closure_span=160., total_cap=ratio-1)
    expected = 1+(ratio-1)*modulation_fraction("combined_opening", temperature=fields["T"].value,
        stress=fields["stress"].value, carbon=fields["carbon"].value, crack=fields["crack"].value,
        transport=transport, mechanics=SimpleNamespace(crack_threshold_stress=threshold))
    np.testing.assert_allclose(gas_modulation("combined_opening", fields, point).value, expected, rtol=1e-14)


def test_opening_gain_gauge_does_not_identify_individual_material_gains():
    stress, damage, loss, closure = .6, .2, .1, .3
    original = (8.-1)*(1.2*stress+2.1*damage+1.1*loss-.4*closure)/9.6
    transformed = (15.-1)*(.6*stress+1.05*damage+.55*loss-.2*closure)/9.6
    np.testing.assert_allclose(original, transformed, rtol=1e-14)


def test_active_clipping_kink_is_rejected_without_silent_smoothing():
    _, layout, raw, gas_point, _, _ = problem()
    fields = {key: layout.observed(key, raw[selection].reshape(layout.blocks[key])) for key, selection in layout.slices.items()}
    point = dict(gas_point)
    point["stress_cap"] = point["stress_gain_per_pa"]*fields["stress"].value.ravel()[0]
    with pytest.raises(ModulationKink):
        gas_modulation("combined_opening", fields, point)


@pytest.mark.parametrize("kind,point", [("constant", {}), ("crack_only", {"gain": 3.}),
    ("carbon_loss_only", {"gain": 2.}), ("stress_only", {"stress_gain_per_pa": 2e-6, "stress_cap": 2.})])
def test_declared_gas_alternatives_require_no_hidden_material_defaults(kind, point):
    _, layout, raw, _, _, _ = problem()
    fields = {key: layout.observed(key, raw[selection].reshape(layout.blocks[key])) for key, selection in layout.slices.items()}
    value = gas_modulation(kind, fields, point)
    assert np.all(value.value >= 1)
    assert value.jacobian.shape[1] == layout.size


def test_new_parameter_derivatives_include_rate_ratio_modulation_and_barrier():
    _, _, raw, gp, hp, build = problem()
    model = build(raw)[0]
    points = {**{"gas.modulation."+key: value for key, value in gp.items()},
              **{"thermal.modulation."+key: value for key, value in hp.items()},
              "kinetic.oxygen.energy_1e5": 1.1, "kinetic.steam.energy_1e5": 1.3,
              "kinetic.oxygen.pressure_order": 1.1, "kinetic.steam.pressure_order": .8,
              "kinetic.carbon_order": 1.2, "reaction.relative_rate": 2.3, "boundary.barrier_strength": .2}
    for name, point in points.items():
        step = 1e-5*max(abs(point), 1e-6)
        plus = build(raw, overrides={name: point+step})[0]
        minus = build(raw, overrides={name: point-step})[0]
        numerical = (plus.matrix-minus.matrix)/(2*step)
        scale = max(float(np.max(np.abs(numerical))), 1e-20)
        np.testing.assert_allclose(model.matrix_jacobian[name]/scale, numerical/scale, rtol=2e-6, atol=2e-8)


def test_complete_observation_derivatives_include_new_physical_and_graph_terms():
    _, layout, raw, _, _, build = problem()
    model, actual, _ = build(raw)
    steps = 2e-5*np.maximum(np.abs(raw), 1.)
    for key in ("pO2", "pH2O", "pProducts"):
        steps[layout.slices[key]] = .004
    reference, numerical = numerical_balance_linearization(lambda y: build(y)[0], raw, layout.coordinate_keys, steps)
    np.testing.assert_array_equal(model.target, reference.target)
    for left, right in zip(actual.column_jacobians, numerical.column_jacobians):
        right = right.toarray()
        scale = max(float(np.max(np.abs(right))), 1e-20)
        np.testing.assert_allclose(left.toarray()/scale, right/scale, rtol=2e-6, atol=2e-8)
