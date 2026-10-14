import numpy as np

from wphid.truth.csic_surface import (
    NozzleSurfaceChemistry,
    csic_surface_rates,
)
from wphid.truth.reusable_nozzle import (
    NozzleGrid,
    NozzleMission,
    NozzleThermalBoundary,
    simulate_reusable_nozzle_mission,
)


def test_surface_rates_separate_oxygen_formation_steam_loss_and_stress_spallation():
    parameters = NozzleSurfaceChemistry()
    common = {
        "temperature": np.array([1900.0]),
        "silica_thickness": np.array([40.0e-6]),
        "underlying_carbon_reaction_rate": np.array([1.0e-5]),
    }
    oxygen_rich = csic_surface_rates(
        oxygen_pressure=np.array([3000.0]),
        steam_pressure=np.array([2.0e5]),
        stress=np.array([50.0e6]),
        parameters=parameters,
        **common,
    )
    steam_rich = csic_surface_rates(
        oxygen_pressure=np.array([500.0]),
        steam_pressure=np.array([1.0e6]),
        stress=np.array([50.0e6]),
        parameters=parameters,
        **common,
    )
    high_stress = csic_surface_rates(
        oxygen_pressure=np.array([3000.0]),
        steam_pressure=np.array([2.0e5]),
        stress=np.array([150.0e6]),
        parameters=parameters,
        **common,
    )

    assert oxygen_rich["passive_formation_rate"][0] > steam_rich["passive_formation_rate"][0]
    assert steam_rich["steam_volatilization_rate"][0] > oxygen_rich["steam_volatilization_rate"][0]
    assert high_stress["stress_spallation_rate"][0] > oxygen_rich["stress_spallation_rate"][0]
    assert 0.0 < oxygen_rich["barrier_factor"][0] < 1.0


def test_surface_extension_adds_bounded_layer_and_monotone_recession_without_changing_default():
    grid = NozzleGrid(
        meridional_cells=9,
        thickness_cells=6,
        time_step=0.02,
        output_interval=10.0,
    )
    mission = NozzleMission(duration=60.0, ramp_time=4.0, shutdown_time=4.0)
    baseline = simulate_reusable_nozzle_mission(
        "baseline", mission=mission, grid=grid
    )
    layered = simulate_reusable_nozzle_mission(
        "layered",
        mission=mission,
        grid=grid,
        thermal_boundary=NozzleThermalBoundary(model="bartz_area_profile"),
        surface_chemistry=NozzleSurfaceChemistry(),
    )

    assert baseline.surface_states == {}
    assert set(layered.surface_states) == {"silica_thickness", "recession_depth"}
    silica = layered.surface_states["silica_thickness"]
    recession = layered.surface_states["recession_depth"]
    assert silica.shape == recession.shape == (layered.t.size, layered.s.size)
    assert np.all((silica >= 0.0) & (silica <= 120.0e-6))
    assert np.all(np.diff(recession, axis=0) >= -1.0e-15)
    assert np.max(recession[-1]) > 0.0
    assert "surface_barrier_factor" in layered.loads
    assert np.all((layered.loads["surface_barrier_factor"] > 0.0) & (layered.loads["surface_barrier_factor"] <= 1.0))
