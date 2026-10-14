from experiments.paper_r1_literature_tps_forward_model import run_experiment
from wphid.truth.reusable_tps import TPSGrid, TPSMission


def test_literature_tps_forward_source_is_qualified(tmp_path):
    grid = TPSGrid(
        axial_cells=9,
        thickness_cells=6,
        time_step=0.04,
        output_interval=10.0,
    )
    base = dict(duration=120.0, peak_time=65.0, oxidant_width=34.0)
    missions = {
        "mild_entry": TPSMission(**base, peak_heat_flux=480_000.0, heat_width=25.0),
        "nominal_entry": TPSMission(**base, peak_heat_flux=600_000.0, heat_width=28.0),
        "hot_entry": TPSMission(**base, peak_heat_flux=720_000.0, heat_width=28.0),
        "long_hot_entry": TPSMission(**base, peak_heat_flux=600_000.0, heat_width=34.0),
    }
    metrics = run_experiment(
        tmp_path / "metrics.json", grid=grid, missions=missions
    )

    assert metrics["aggregate"]["all_checks_pass"] is True
    assert set(metrics["cases"]) == {
        "mild_entry",
        "nominal_entry",
        "hot_entry",
        "long_hot_entry",
    }
    assert metrics["cases"]["hot_entry"]["maximum_end_carbon_loss"] > metrics[
        "cases"
    ]["mild_entry"]["maximum_end_carbon_loss"]
