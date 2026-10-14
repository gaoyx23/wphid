import numpy as np

from wphid.truth.reusable_cmc import (
    reusable_cmc_evaluation_truth,
    simulate_reusable_cmc_mission,
)
from wphid.truth.reusable_tps import TPSGrid, TPSMission


_GRID = TPSGrid(
    axial_cells=9,
    thickness_cells=6,
    time_step=0.04,
    output_interval=10.0,
)
_MISSION = TPSMission(
    duration=120.0,
    peak_time=65.0,
    heat_width=28.0,
    oxidant_width=34.0,
)


def test_multifield_model_exports_six_coupled_fields():
    trajectory = simulate_reusable_cmc_mission(
        "short-entry", mission=_MISSION, grid=_GRID
    )

    assert set(trajectory.states) == {
        "T",
        "pO2",
        "pCOx",
        "carbon",
        "crack",
        "stress",
    }
    assert trajectory.states["pO2"].max() > 0.0
    assert trajectory.states["pCOx"].max() > 0.0
    assert trajectory.states["carbon"][-1].min() < 1.0
    assert trajectory.states["crack"][-1].max() > trajectory.states["crack"][0].max()
    assert trajectory.states["stress"].max() > 1.0e6


def test_hotter_natural_mission_increases_carbon_loss_and_cracking():
    mild = simulate_reusable_cmc_mission(
        "mild", mission=_MISSION, grid=_GRID
    )
    hot = simulate_reusable_cmc_mission(
        "hot",
        mission=TPSMission(
            duration=120.0,
            peak_time=65.0,
            heat_width=32.0,
            oxidant_width=38.0,
            peak_heat_flux=720_000.0,
        ),
        grid=_GRID,
    )

    assert hot.states["T"].max() > mild.states["T"].max()
    assert hot.states["carbon"][-1].min() < mild.states["carbon"][-1].min()
    assert hot.states["crack"][-1].max() > mild.states["crack"][-1].max()


def test_truth_graph_contains_transport_chemistry_and_mechanics():
    truth = reusable_cmc_evaluation_truth()

    assert len(truth["fields"]) == 6
    assert set(truth["primary_hidden_mechanisms"]) == {
        "carbon_oxidation",
        "stress_assisted_cracking",
        "oxidation_assisted_cracking",
        "crack_controlled_transport",
    }
    assert "pCOx" in truth["topology"]["carbon_oxidation"]


def test_existing_damage_changes_gas_transport_and_stress():
    fresh = simulate_reusable_cmc_mission(
        "fresh", mission=_MISSION, grid=_GRID
    )
    aged = simulate_reusable_cmc_mission(
        "aged",
        mission=_MISSION,
        grid=_GRID,
        initial_carbon_fraction=0.92,
        initial_crack_damage=0.12,
    )

    assert not np.allclose(fresh.states["pO2"], aged.states["pO2"])
    assert not np.allclose(fresh.states["stress"], aged.states["stress"])
