from experiments.paper_r97_layered_surface_confirmation import _missions


def test_r97_confirmation_changes_only_gas_composition():
    missions = _missions()
    reference = missions["nominal"]
    for mission in missions.values():
        assert mission.duration == reference.duration
        assert mission.chamber_pressure == reference.chamber_pressure
        assert mission.heat_flux_throat == reference.heat_flux_throat
        assert mission.ramp_time == reference.ramp_time
        assert mission.shutdown_time == reference.shutdown_time
    assert missions["oxygen_richer_matched"].residual_oxygen_fraction > reference.residual_oxygen_fraction
    assert missions["high_steam_matched"].steam_mole_fraction > reference.steam_mole_fraction
