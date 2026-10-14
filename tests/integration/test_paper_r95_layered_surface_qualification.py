from experiments.paper_r95_layered_csic_surface_qualification import _cases


def test_r95_cases_are_passive_bounded_hot_fire_variants():
    cases = _cases()
    assert set(cases) == {
        "oxygen_richer_750s",
        "nominal_900s",
        "high_thermal_750s",
        "high_steam_900s",
    }
    assert all(0.0 < mission.chamber_pressure <= 4.0e6 for mission in cases.values())
    assert all(0.0 < mission.heat_flux_throat <= 650_000.0 for mission in cases.values())
    assert all(0.0 <= mission.residual_oxygen_fraction <= 1.0e-3 for mission in cases.values())
    assert all(0.0 < mission.steam_mole_fraction < 1.0 for mission in cases.values())
