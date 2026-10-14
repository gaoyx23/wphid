import numpy as np

from wphid.truth.reusable_nozzle import (
    bartz_scaled_heat_flux,
    NozzleGeometry,
    NozzleGrid,
    NozzleMission,
    NozzleThermalBoundary,
    NozzleTransport,
    nozzle_heat_flux_shape,
    reusable_nozzle_evaluation_truth,
    simulate_reusable_nozzle_mission,
)


_GRID = NozzleGrid(
    meridional_cells=9,
    thickness_cells=6,
    time_step=0.02,
    output_interval=10.0,
)
_MISSION = NozzleMission(
    duration=60.0,
    ramp_time=4.0,
    shutdown_time=4.0,
)


def test_hm7_class_geometry_matches_public_envelope_and_mass():
    geometry = NozzleGeometry()
    z = np.linspace(0.0, geometry.length, 101)
    radius = geometry.inner_radius(z)

    np.testing.assert_allclose(geometry.length, 0.988)
    np.testing.assert_allclose(2.0 * geometry.exit_radius, 0.938)
    np.testing.assert_allclose(
        (geometry.exit_radius / geometry.throat_radius) ** 2,
        geometry.expansion_ratio,
    )
    assert np.all(np.diff(radius) > 0.0)
    assert abs(geometry.shell_mass() - geometry.reference_mass) / geometry.reference_mass < 0.01


def test_bartz_scaling_keeps_pressure_and_heat_flux_physically_coupled():
    reference = bartz_scaled_heat_flux(3.6e6)
    lower = bartz_scaled_heat_flux(3.24e6)
    higher = bartz_scaled_heat_flux(3.96e6)

    np.testing.assert_allclose(reference, 575_000.0)
    assert lower < reference < higher
    np.testing.assert_allclose(higher / lower, (3.96 / 3.24) ** 0.8)


def test_bartz_area_profile_is_optional_and_legacy_default_is_unchanged():
    geometry = NozzleGeometry()
    z = np.linspace(0.0, geometry.length, 51)
    legacy = nozzle_heat_flux_shape(z, geometry=geometry)
    bartz = nozzle_heat_flux_shape(
        z,
        geometry=geometry,
        boundary=NozzleThermalBoundary(model="bartz_area_profile"),
    )

    np.testing.assert_allclose(legacy[0], 1.0)
    np.testing.assert_allclose(bartz[0], 1.0)
    assert np.all(np.diff(legacy) < 0.0)
    assert np.all(np.diff(bartz) < 0.0)
    assert bartz[-1] < legacy[-1] / 4.0


def test_bartz_area_profile_changes_only_opted_in_future_truth():
    legacy = simulate_reusable_nozzle_mission(
        "legacy-boundary", mission=_MISSION, grid=_GRID
    )
    upgraded = simulate_reusable_nozzle_mission(
        "bartz-boundary",
        mission=_MISSION,
        grid=_GRID,
        thermal_boundary=NozzleThermalBoundary(model="bartz_area_profile"),
    )

    assert upgraded.loads["surface_heat_flux"][:, -1].max() < (
        legacy.loads["surface_heat_flux"][:, -1].max() / 4.0
    )
    assert upgraded.states["T"][:, -1].max() < legacy.states["T"][:, -1].max()


def test_nozzle_model_exports_geometry_resolved_seven_field_truth():
    trajectory = simulate_reusable_nozzle_mission(
        "short-hot-fire", mission=_MISSION, grid=_GRID
    )

    assert set(trajectory.states) == {
        "T", "pO2", "pH2O", "pProducts", "carbon", "crack", "stress"
    }
    assert trajectory.z.shape == trajectory.r.shape == (9, 6)
    assert trajectory.states["pH2O"].max() > trajectory.states["pO2"].max()
    assert trajectory.states["pProducts"].max() > 0.0
    assert trajectory.states["carbon"][-1].min() < 1.0
    assert trajectory.states["crack"][-1].max() > trajectory.states["crack"][0].max()
    assert trajectory.states["stress"].max() > 1.0e6
    assert np.ptp(trajectory.loads["gas_pressure"].max(axis=0)) > 1.0e5


def test_hotter_higher_pressure_fire_has_stronger_response():
    baseline = simulate_reusable_nozzle_mission("baseline", mission=_MISSION, grid=_GRID)
    severe = simulate_reusable_nozzle_mission(
        "severe",
        mission=NozzleMission(
            duration=60.0,
            ramp_time=4.0,
            shutdown_time=4.0,
            chamber_pressure=4.0e6,
            heat_flux_throat=700_000.0,
            steam_mole_fraction=0.78,
        ),
        grid=_GRID,
    )

    assert severe.states["T"].max() > baseline.states["T"].max()
    assert severe.states["carbon"][-1].min() < baseline.states["carbon"][-1].min()
    assert severe.states["crack"][-1].max() > baseline.states["crack"][-1].max()
    assert severe.states["stress"].max() > baseline.states["stress"].max()


def test_truth_graph_separates_oxygen_and_steam_mechanisms():
    truth = reusable_nozzle_evaluation_truth()

    assert len(truth["fields"]) == 7
    assert set(truth["primary_hidden_mechanisms"]) == {
        "oxygen_carbon_oxidation",
        "steam_corrosion",
        "stress_assisted_cracking",
        "chemistry_assisted_cracking",
        "crack_controlled_transport",
    }
    assert "pH2O" in truth["topology"]["steam_corrosion"]
    assert "pO2" in truth["topology"]["oxygen_carbon_oxidation"]


def test_reflight_inherits_full_carbon_and_crack_fields():
    first = simulate_reusable_nozzle_mission(
        "cycle-1", mission=_MISSION, grid=_GRID
    )
    second = simulate_reusable_nozzle_mission(
        "cycle-2",
        mission=_MISSION,
        grid=_GRID,
        initial_carbon_fraction=first.states["carbon"][-1],
        initial_crack_damage=first.states["crack"][-1],
    )

    np.testing.assert_allclose(
        second.states["carbon"][0], first.states["carbon"][-1]
    )
    np.testing.assert_allclose(
        second.states["crack"][0], first.states["crack"][-1]
    )


def test_postfire_cooldown_inherits_all_dynamic_fields_and_removes_hot_gas_load():
    first = simulate_reusable_nozzle_mission(
        "hot-fire", mission=_MISSION, grid=_GRID
    )
    inherited = {
        name: first.states[name][-1]
        for name in ("T", "pO2", "pH2O", "pProducts", "carbon", "crack")
    }
    cooldown = simulate_reusable_nozzle_mission(
        "cooldown",
        mission=NozzleMission(
            duration=40.0,
            chamber_pressure=0.0,
            heat_flux_throat=0.0,
            steam_mole_fraction=0.0,
            residual_oxygen_fraction=0.0,
            outer_oxygen_pressure=21_000.0,
            ramp_time=2.0,
            shutdown_time=2.0,
        ),
        grid=_GRID,
        initial_states=inherited,
    )

    for name, values in inherited.items():
        np.testing.assert_allclose(cooldown.states[name][0], values)
    assert np.max(cooldown.loads["gas_pressure"]) == 0.0
    assert np.max(cooldown.loads["surface_heat_flux"]) == 0.0


def test_optional_directional_forchheimer_transport_preserves_default_contract():
    reference = simulate_reusable_nozzle_mission(
        "transport-default", mission=_MISSION, grid=_GRID
    )
    explicit_default = simulate_reusable_nozzle_mission(
        "transport-explicit-default",
        mission=_MISSION,
        grid=_GRID,
        transport=NozzleTransport(
            meridional_to_through_diffusivity_ratio=1.0,
            forchheimer_inverse_flux_scale=0.0,
        ),
    )

    for name in reference.states:
        np.testing.assert_allclose(
            reference.states[name], explicit_default.states[name], rtol=0.0, atol=0.0
        )


def test_full_coupled_transport_options_propagate_beyond_gas_fields():
    isotropic = simulate_reusable_nozzle_mission(
        "coupled-isotropic", mission=_MISSION, grid=_GRID
    )
    nonlinear_directional = simulate_reusable_nozzle_mission(
        "coupled-nonlinear-directional",
        mission=_MISSION,
        grid=_GRID,
        transport=NozzleTransport(
            meridional_to_through_diffusivity_ratio=3.0,
            forchheimer_inverse_flux_scale=1.0,
        ),
    )

    assert not np.array_equal(
        nonlinear_directional.states["pH2O"], isotropic.states["pH2O"]
    )
    assert not np.array_equal(
        nonlinear_directional.states["carbon"], isotropic.states["carbon"]
    )
    assert not np.array_equal(
        nonlinear_directional.states["crack"], isotropic.states["crack"]
    )
