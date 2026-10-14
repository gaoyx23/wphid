import numpy as np

from wphid.engineering_observations import (
    project_engineering_observations,
    sample_engineering_observations,
)
from wphid.truth.reusable_nozzle import NozzleGrid, NozzleMission, simulate_reusable_nozzle_mission


def _trajectory():
    return simulate_reusable_nozzle_mission(
        "engineering-observation-test",
        mission=NozzleMission(duration=30.0, ramp_time=3.0, shutdown_time=3.0),
        grid=NozzleGrid(
            meridional_cells=9,
            thickness_cells=6,
            time_step=0.02,
            output_interval=5.0,
        ),
    )


def test_instrumented_rig_is_a_strict_measurement_superset():
    trajectory = _trajectory()
    standard = project_engineering_observations(
        trajectory, tier="standard_ground_test", time_samples=5
    )
    rig = project_engineering_observations(
        trajectory, tier="instrumented_research_rig", time_samples=5
    )

    assert rig.values.size > standard.values.size
    assert rig.labels[: standard.values.size] == standard.labels
    assert set(standard.blocks).issubset(set(rig.blocks))
    assert np.all(standard.noise_std > 0.0)

    acoustic = project_engineering_observations(
        trajectory,
        tier="instrumented_rig_with_acoustic_emission",
        time_samples=5,
    )
    assert acoustic.values.size > rig.values.size
    assert acoustic.labels[: rig.values.size] == rig.labels
    assert "distributed_acoustic_crack_rate_proxy" in acoustic.blocks


def test_engineering_noise_and_missingness_are_seed_reproducible():
    trajectory = _trajectory()
    first, first_mask = sample_engineering_observations(
        trajectory,
        tier="standard_ground_test",
        time_samples=5,
        seed=71001,
        missing_fraction=0.1,
    )
    second, second_mask = sample_engineering_observations(
        trajectory,
        tier="standard_ground_test",
        time_samples=5,
        seed=71001,
        missing_fraction=0.1,
    )

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert not np.all(first_mask)
