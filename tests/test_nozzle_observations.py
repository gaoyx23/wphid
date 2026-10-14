import numpy as np

from wphid.observations import (
    NOISE_MULTIPLIERS,
    OBSERVATION_TIERS,
    observation_packet_summary,
    reconstruct_nozzle_channel,
    reconstruct_nozzle_channel_on_grid,
    sample_nozzle_observations,
)
from wphid.truth.reusable_nozzle import (
    NozzleGrid,
    NozzleMission,
    simulate_reusable_nozzle_mission,
)


_TRAJECTORY = simulate_reusable_nozzle_mission(
    "observation-test",
    mission=NozzleMission(duration=40.0, ramp_time=3.0, shutdown_time=3.0),
    grid=NozzleGrid(
        meridional_cells=9,
        thickness_cells=6,
        time_step=0.02,
        output_interval=2.0,
    ),
)


def test_observation_tiers_add_channels_monotonically():
    limited = set(OBSERVATION_TIERS["limited"])
    intermediate = set(OBSERVATION_TIERS["intermediate"])
    sufficient = set(OBSERVATION_TIERS["sufficient"])

    assert limited == {"T", "crack"}
    assert limited < intermediate < sufficient
    assert sufficient == set(_TRAJECTORY.states)
    assert set(NOISE_MULTIPLIERS) == {"clean", "nominal", "high"}


def test_sampling_is_reproducible_sparse_asynchronous_and_missing():
    first = sample_nozzle_observations(
        _TRAJECTORY, tier="sufficient", noise_level="nominal", seed=52101
    )
    second = sample_nozzle_observations(
        _TRAJECTORY, tier="sufficient", noise_level="nominal", seed=52101
    )

    for name in first.channels:
        one, two = first.channels[name], second.channels[name]
        np.testing.assert_array_equal(one.retained, two.retained)
        np.testing.assert_allclose(one.values, two.values, equal_nan=True)
        assert np.any(~one.retained)
    clocks = {tuple(np.round(channel.times, 8)) for channel in first.channels.values()}
    assert len(clocks) == len(first.channels)


def test_clean_reconstruction_is_finite_and_preserves_bounds():
    packet = sample_nozzle_observations(
        _TRAJECTORY, tier="intermediate", noise_level="clean", seed=52102
    )

    for name in packet.channels:
        reconstructed, diagnostics = reconstruct_nozzle_channel(
            packet, name, _TRAJECTORY
        )
        assert np.all(np.isfinite(reconstructed))
        assert diagnostics["spatial_sensor_fraction"] <= 1.0
        assert diagnostics["nominal_time_fraction"] <= 1.0
        assert diagnostics["field_nrmse"] < 0.30
    assert np.all((reconstructed >= 0.0) if name != "T" else (reconstructed > 0.0))


def test_packet_summary_contains_observations_not_clean_fields():
    packet = sample_nozzle_observations(
        _TRAJECTORY, tier="limited", noise_level="high", seed=52103
    )
    summary = observation_packet_summary(packet)

    assert summary["tier"] == "limited"
    assert set(summary["channels"]) == {"T", "crack"}
    assert all("noise_sigma" in row for row in summary["channels"].values())
    assert "states" not in summary


def test_irreversible_state_reconstruction_can_enforce_physics_monotonicity():
    packet = sample_nozzle_observations(
        _TRAJECTORY, tier="sufficient", noise_level="nominal", seed=52104
    )
    constraints = {"carbon": "nonincreasing", "crack": "nondecreasing"}

    for name, constraint in constraints.items():
        reconstructed, diagnostics = reconstruct_nozzle_channel_on_grid(
            packet.channels[name],
            target_t=_TRAJECTORY.t,
            target_s=_TRAJECTORY.s,
            target_x=_TRAJECTORY.x,
            temporal_constraint=constraint,
        )
        differences = np.diff(reconstructed, axis=0)
        if constraint == "nonincreasing":
            assert np.max(differences) <= 1e-12
        else:
            assert np.min(differences) >= -1e-12
        assert diagnostics["temporal_constraint"] == constraint


def test_optional_meridional_reconstructions_are_finite_and_reported():
    packet = sample_nozzle_observations(
        _TRAJECTORY, tier="sufficient", noise_level="nominal", seed=52105
    )
    channel = packet.channels["pProducts"]

    for method in ("pchip_meridional", "noise_calibrated_spline_meridional"):
        reconstructed, diagnostics = reconstruct_nozzle_channel_on_grid(
            channel,
            target_t=_TRAJECTORY.t,
            target_s=_TRAJECTORY.s,
            target_x=_TRAJECTORY.x,
            spatial_method=method,
        )
        assert np.all(np.isfinite(reconstructed))
        assert reconstructed.shape == _TRAJECTORY.states["pProducts"].shape
        assert diagnostics["spatial_method"] == method
        assert diagnostics["spatial_smoothing_multiplier"] == 1.0
