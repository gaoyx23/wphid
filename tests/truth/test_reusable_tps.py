import numpy as np
import pytest

from wphid.truth.reusable_tps import (
    TPSGrid,
    TPSMission,
    reusable_tps_evaluation_truth,
    simulate_reusable_tps_mission,
)


_FAST_GRID = TPSGrid(
    axial_cells=9,
    thickness_cells=6,
    time_step=0.04,
    output_interval=10.0,
)
_FAST_MISSION = TPSMission(
    duration=120.0,
    peak_time=65.0,
    heat_width=28.0,
    oxidant_width=34.0,
)


def test_forward_model_generates_coupled_physical_fields():
    trajectory = simulate_reusable_tps_mission(
        "short-entry", mission=_FAST_MISSION, grid=_FAST_GRID
    )

    assert set(trajectory.states) == {"T", "O", "D"}
    assert trajectory.states["T"].shape == (13, 9, 6)
    assert trajectory.states["T"].max() > trajectory.states["T"][0].max()
    assert trajectory.states["O"].max() > 0.0
    assert trajectory.states["D"].max() > 0.0
    assert trajectory.states["T"][:, :, 0].max() > trajectory.states["T"][:, :, -1].max()


def test_more_severe_natural_mission_increases_damage_without_state_perturbation():
    nominal = simulate_reusable_tps_mission(
        "nominal", mission=_FAST_MISSION, grid=_FAST_GRID
    )
    severe = simulate_reusable_tps_mission(
        "hotter",
        mission=TPSMission(
            duration=120.0,
            peak_time=65.0,
            heat_width=32.0,
            oxidant_width=38.0,
            peak_heat_flux=720_000.0,
        ),
        grid=_FAST_GRID,
    )

    assert severe.states["T"].max() > nominal.states["T"].max()
    assert severe.states["D"][-1].max() > nominal.states["D"][-1].max()


def test_damage_history_changes_transport_and_future_degradation():
    fresh = simulate_reusable_tps_mission(
        "fresh", mission=_FAST_MISSION, grid=_FAST_GRID
    )
    aged = simulate_reusable_tps_mission(
        "aged", mission=_FAST_MISSION, grid=_FAST_GRID, initial_damage=0.12
    )

    assert aged.states["D"][0].min() == pytest.approx(0.12)
    assert not np.allclose(aged.states["O"], fresh.states["O"])
    assert not np.allclose(aged.states["T"], fresh.states["T"])


def test_truth_metadata_names_the_complete_shared_nonlinear_term():
    truth = reusable_tps_evaluation_truth()

    assert truth["shared_hidden_mechanism"] == "A*exp(-Ea/(R*T))*O^m*(1-D)^q"
    assert set(truth["destinations"]) == {"temperature", "oxidant", "damage"}
    assert truth["oxidation"]["activation_energy"] == 123_000.0


def test_unstable_explicit_grid_is_rejected():
    with pytest.raises(ValueError, match="stability"):
        simulate_reusable_tps_mission(
            "unstable",
            mission=_FAST_MISSION,
            grid=TPSGrid(
                axial_cells=9,
                thickness_cells=12,
                time_step=0.2,
                output_interval=10.0,
            ),
        )
