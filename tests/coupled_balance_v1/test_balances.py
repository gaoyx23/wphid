"""Independent balance identities, declared boundaries and outer derivatives."""

from dataclasses import replace

import numpy as np
import pytest

from wphid.coupled_balance_v1 import (
    BoundaryInputs, Modulation, RatePair, ShellMesh, TransportShape, WeakProjector,
    arrhenius_pair, assemble_balances, heat_primitive, temporal_tests,
)
from wphid.coupled_balance_v1.assembler import COLUMNS, FIELDS, PRESSURES
from wphid.coupled_balance_v1.operators import SIDES, attenuated_flux


def flat_mesh(ns=2, nx=3, ds=1., dx=1.):
    return ShellMesh(
        np.full((ns, nx), ds*dx), np.full((ns-1, nx), dx), np.full((ns, nx-1), ds),
        np.full((ns-1, nx), ds), np.full((ns, nx-1), dx),
        {"inner": np.full(ns, ds), "outer": np.full(ns, ds),
         "inlet": np.full(nx, dx), "outlet": np.full(nx, dx)},
    )


def sealed_boundary():
    return BoundaryInputs({}, {}, frozenset((k, s) for k in PRESSURES for s in SIDES),
                          {}, {}, frozenset(SIDES))


PARAMETERS = {
    "kinetic.oxygen_E_1e5": 1.1, "kinetic.steam_E_1e5": 1.3,
    "kinetic.oxygen_order": 1.1, "kinetic.steam_order": .8, "kinetic.carbon_order": 1.2,
    "reaction.relative_rate": 1.4, "gas.direction_ratio": 1.3, "gas.steam_ratio": .8,
    "gas.product_ratio": 1.2, "gas.forchheimer_product": .005,
    "thermal.direction_ratio": 1.7, "gas.carbon_gain": .7,
    "thermal.carbon_gain": .2, "surface.barrier_gain": .3,
}


def example_system(parameters=None):
    p = PARAMETERS if parameters is None else parameters
    t = np.arange(9.)
    mesh = flat_mesh()
    tt, ii, jj = np.meshgrid(t, np.arange(2.), np.arange(3.), indexing="ij")
    states = {
        "carbon": .9-.001*tt+.002*ii+.003*jj,
        "T": 500+10*tt+3*ii+2*jj,
        "pO2": 2e4+200*tt+350*ii+250*jj,
        "pH2O": 6e5+800*tt+1200*ii+2000*jj,
        "pProducts": 500+10*tt+5*ii+6*jj,
    }
    spatial = np.stack((np.ones(mesh.shape), np.array([[1., -.4, .2], [.3, .8, -.1]])))
    projector = WeakProjector(mesh, temporal_tests(t, window_samples=5, stride_samples=2), spatial)
    rates = arrhenius_pair(states, oxygen_energy=p["kinetic.oxygen_E_1e5"]*1e5,
                           steam_energy=p["kinetic.steam_E_1e5"]*1e5,
                           oxygen_order=p["kinetic.oxygen_order"], steam_order=p["kinetic.steam_order"],
                           carbon_order=p["kinetic.carbon_order"])
    carbon_loss = 1-states["carbon"]
    gas = Modulation(1+p["gas.carbon_gain"]*carbon_loss, {"gas.carbon_gain": carbon_loss})
    thermal = Modulation(1-p["thermal.carbon_gain"]*carbon_loss, {"thermal.carbon_gain": -carbon_loss})
    external = {("pO2", "inner"): np.full((9, 2), 3e4),
                ("pH2O", "inner"): np.full((9, 2), 7e5),
                ("pProducts", "inner"): np.zeros((9, 2)),
                ("pO2", "outer"): np.full((9, 2), 2.2e4)}
    barrier_basis = np.broadcast_to(.001*t[:, None], (9, 2))
    barriers = {key: Modulation(1+p["surface.barrier_gain"]*barrier_basis,
                               {"surface.barrier_gain": barrier_basis}) for key in external}
    sealed = frozenset((k, s) for k in PRESSURES for s in SIDES if (k, s) not in external)
    boundary = BoundaryInputs(external, barriers, sealed,
                              {"inner": np.broadcast_to((50+5*t)[:, None], (9, 2))},
                              {"outer": np.full((9, 2), 300.)}, frozenset(("inlet", "outlet")))
    system = assemble_balances(
        projector, states, rates, relative_rate=p["reaction.relative_rate"],
        transport=TransportShape(p["gas.direction_ratio"], p["gas.steam_ratio"],
                                 p["gas.product_ratio"], p["gas.forchheimer_product"]),
        gas_modulation=gas, thermal_modulation=thermal,
        thermal_direction_ratio=p["thermal.direction_ratio"], heat_knots=(400., 1000.), boundary=boundary,
    )
    return system, projector, states, rates, boundary


def test_face_contraction_equals_control_volume_balance_for_arbitrary_tests():
    mesh = flat_mesh(ns=2, nx=3, ds=.7, dx=.4)
    t = np.linspace(0., 2., 9)
    phi = np.array([[[1.2, -.8, 2.1], [.3, .7, -1.]]])
    projector = WeakProjector(mesh, temporal_tests(t, window_samples=9, stride_samples=1), phi)
    js = np.broadcast_to(np.array([[.4, -.2, .7]]), (9, 1, 3))
    jx = np.broadcast_to(np.array([[.5, -.3], [.2, .8]]), (9, 2, 2))
    cell_rate = np.zeros((9, 2, 3))
    cell_rate[:, :-1] -= js*mesh.area_s/mesh.volume[:-1]
    cell_rate[:, 1:] += js*mesh.area_s/mesh.volume[1:]
    cell_rate[:, :, :-1] -= jx*mesh.area_x/mesh.volume[:, :-1]
    cell_rate[:, :, 1:] += jx*mesh.area_x/mesh.volume[:, 1:]
    np.testing.assert_allclose(projector.flux(js, jx), projector.source(cell_rate), atol=1e-12)


def test_closed_internal_flux_has_zero_constant_test_mass_change():
    mesh = flat_mesh()
    projector = WeakProjector(mesh, temporal_tests(np.arange(9.), window_samples=5, stride_samples=2),
                              np.ones((1, *mesh.shape)))
    js = np.broadcast_to(np.array([[.4, -.2, .7]]), (9, 1, 3))
    jx = np.broadcast_to(np.array([[.5, -.3], [.2, .8]]), (9, 2, 2))
    np.testing.assert_array_equal(projector.flux(js, jx), np.zeros(3))


def test_open_parabolic_pressure_satisfies_independent_analytic_pde():
    t = np.linspace(0., 2., 9)
    mesh = flat_mesh(ns=1, nx=3, dx=1/3)
    projector = WeakProjector(mesh, temporal_tests(t, window_samples=9, stride_samples=1),
                              np.array([[[1.2, -.8, 2.1]]]))
    x = (np.arange(3)+.5)/3
    shape = (9, 1, 3)
    states = {key: np.ones(shape) for key in FIELDS}
    states["pO2"] = 3+2*t[:, None, None]+x[None, None, :]**2
    external = {("pO2", "outer"): states["pO2"][:, :, -1]+2}
    sealed = frozenset((k, s) for k in PRESSURES for s in SIDES if (k, s) not in external)
    boundary = BoundaryInputs(external, {("pO2", "outer"): Modulation(np.ones((9, 1)))}, sealed,
                              {}, {}, frozenset(SIDES))
    system = assemble_balances(projector, states, RatePair(np.zeros((2, *shape))), relative_rate=1.,
        transport=TransportShape(1., 1., 1., 0.), gas_modulation=Modulation(np.ones(shape)),
        thermal_modulation=Modulation(np.ones(shape)), thermal_direction_ratio=1., heat_knots=(0., 10.), boundary=boundary)
    coefficients = np.zeros(11)
    coefficients[COLUMNS.index("gas_scale")] = 1.
    coefficients[COLUMNS.index("surface_transfer")] = 1.
    np.testing.assert_allclose(system.residual(coefficients), 0., atol=1e-12)


def test_affine_unknown_heat_capacity_satisfies_analytic_energy_balance():
    t = np.linspace(0., 2., 9)
    mesh = flat_mesh(ns=1, nx=1)
    projector = WeakProjector(mesh, temporal_tests(t, window_samples=9, stride_samples=1), np.ones((1, 1, 1)))
    shape = (9, 1, 1)
    states = {key: np.ones(shape) for key in FIELDS}
    states["T"] = (2+t)[:, None, None]
    # T'=1 and cp/cp0=1+.3*T/10, giving an independently specified heat input.
    boundary = replace(sealed_boundary(), heat_inward={"inner": (1+.03*(2+t))[:, None]},
                       sealed_heat=frozenset(("outer", "inlet", "outlet")))
    system = assemble_balances(projector, states, RatePair(np.zeros((2, *shape))), relative_rate=1.,
        transport=TransportShape(1., 1., 1., 0.), gas_modulation=Modulation(np.ones(shape)),
        thermal_modulation=Modulation(np.ones(shape)), thermal_direction_ratio=1., heat_knots=(0., 10.), boundary=boundary)
    coefficients = np.zeros(11)
    coefficients[COLUMNS.index("heat_input_scale")] = 1.
    coefficients[COLUMNS.index("heat_capacity_contrast")] = .3
    np.testing.assert_allclose(system.residual(coefficients), 0., atol=1e-12)


def test_heat_primitive_is_continuous_and_its_derivative_is_the_declared_fraction():
    temperatures = np.array([-1., 0., 3., 10., 11.])
    primitive, fraction = heat_primitive(temperatures, (0., 10.))
    np.testing.assert_allclose(primitive, [0., 0., .45, 5., 6.])
    h = 1e-6
    numerical = (heat_primitive(temperatures+h, (0., 10.))[0]-
                 heat_primitive(temperatures-h, (0., 10.))[0])/(2*h)
    np.testing.assert_allclose(numerical, fraction, atol=1e-7)


def test_euler_storage_chain_rule_defect_is_retained():
    before, increment, contrast = 3., .25, .3
    left = before+contrast*heat_primitive(before, (0., 10.))[0]
    right = before+increment+contrast*heat_primitive(before+increment, (0., 10.))[0]
    euler = (1+contrast*before/10)*increment
    np.testing.assert_allclose(right-left-euler, contrast*increment**2/20, atol=1e-14)
    assert right-left-euler > 0


def test_reference_scale_product_representation_satisfies_constitutive_equation():
    base = np.array([-3., -.3, 0., 1.1, 4.])
    scale, alpha = 2.3, .2
    physical = scale*attenuated_flux(base, alpha*scale)[0]
    np.testing.assert_allclose(physical+alpha*np.abs(physical)*physical, scale*base, atol=1e-12)


def test_constitutive_derivatives_agree_with_independent_central_differences():
    base, product, step = np.array([-3., -.3, 0., 1.1, 4.]), .4, 1e-6
    _, dv, db = attenuated_flux(base, product)
    numerical_v = (attenuated_flux(base+step, product)[0]-attenuated_flux(base-step, product)[0])/(2*step)
    numerical_b = (attenuated_flux(base, product+step)[0]-attenuated_flux(base, product-step)[0])/(2*step)
    np.testing.assert_allclose(dv, numerical_v, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(db, numerical_b, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("key", tuple(PARAMETERS))
def test_complete_outer_derivatives_include_transport_and_boundary_terms(key):
    central = example_system()[0]
    plus, minus = dict(PARAMETERS), dict(PARAMETERS)
    step = 1e-6
    plus[key] += step
    minus[key] -= step
    numerical = (example_system(plus)[0].matrix-example_system(minus)[0].matrix)/(2*step)
    actual = central.matrix_jacobian[key]
    scale = max(float(np.max(np.abs(actual))), 1e-100)
    np.testing.assert_allclose(numerical, actual, rtol=2e-5, atol=2e-8*scale)


def test_missing_pressure_boundary_cannot_be_silently_assumed_sealed():
    _, projector, states, rates, boundary = example_system()
    incomplete = replace(boundary, sealed_pressure=boundary.sealed_pressure-frozenset((("pProducts", "outer"),)))
    with pytest.raises(ValueError, match="explicitly"):
        assemble_balances(projector, states, rates, relative_rate=1.4,
            transport=TransportShape(1.3, .8, 1.2, .005), gas_modulation=Modulation(np.ones(projector.state_shape)),
            thermal_modulation=Modulation(np.ones(projector.state_shape)), thermal_direction_ratio=1.,
            heat_knots=(400., 1000.), boundary=incomplete)


def test_complete_source_storage_and_nuisance_columns_are_retained():
    system = example_system()[0]
    assert system.column_keys == COLUMNS
    assert system.matrix.shape == (30, 11)
    assert len(system.row_keys) == 30
    assert set(row[0] for row in system.row_keys) == set(FIELDS)
    assert set(system.matrix_jacobian) == set(PARAMETERS)


def test_nonpositive_cell_measure_is_not_accepted():
    mesh = flat_mesh()
    with pytest.raises(ValueError, match="positive"):
        replace(mesh, volume=np.zeros(mesh.shape))
