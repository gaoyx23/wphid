"""Passive sparse observation protocols for geometry-resolved nozzle truth."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, UnivariateSpline

from .truth.reusable_nozzle import NozzleFieldTrajectory


NOISE_MULTIPLIERS = MappingProxyType({"clean": 0.0, "nominal": 1.0, "high": 2.0})

# One-multiplier measurement uncertainty. Relative entries use each mission's
# clean dynamic range, with an absolute floor for nearly constant channels.
CHANNEL_NOISE = MappingProxyType({
    "T": (0.0, 5.0),
    "pO2": (0.02, 20.0),
    "pH2O": (0.02, 500.0),
    "pProducts": (0.05, 5.0),
    "carbon": (0.0, 0.002),
    "crack": (0.0, 0.0015),
    "stress": (0.01, 0.5e6),
})


@dataclass(frozen=True)
class ChannelSpec:
    meridional_sensors: int
    thickness_sensors: int
    time_samples: int
    missing_fraction: float


@dataclass(frozen=True)
class NozzleObservationChannel:
    name: str
    times: np.ndarray
    s_indices: np.ndarray
    x_indices: np.ndarray
    values: np.ndarray
    retained: np.ndarray
    noise_sigma: float

    def __post_init__(self) -> None:
        times = _readonly(self.times)
        s_indices = _readonly(self.s_indices, dtype=int)
        x_indices = _readonly(self.x_indices, dtype=int)
        values = _readonly(self.values)
        retained = _readonly(self.retained, dtype=bool)
        expected = (times.size, s_indices.size, x_indices.size)
        if values.shape != expected or retained.shape != expected:
            raise ValueError(f"channel arrays must have shape {expected}")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("channel times must be strictly increasing")
        if not np.array_equal(np.isfinite(values), retained):
            raise ValueError("finite observed values must exactly match retained mask")
        if self.noise_sigma < 0.0 or not np.isfinite(self.noise_sigma):
            raise ValueError("noise_sigma must be finite and non-negative")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "s_indices", s_indices)
        object.__setattr__(self, "x_indices", x_indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "retained", retained)


@dataclass(frozen=True)
class NozzleObservationPacket:
    condition_id: str
    tier: str
    noise_level: str
    channels: Mapping[str, NozzleObservationChannel]
    seed: int

    def __post_init__(self) -> None:
        if self.tier not in OBSERVATION_TIERS:
            raise ValueError(f"unknown observation tier {self.tier!r}")
        if self.noise_level not in NOISE_MULTIPLIERS:
            raise ValueError(f"unknown noise level {self.noise_level!r}")
        expected = set(OBSERVATION_TIERS[self.tier])
        if set(self.channels) != expected:
            raise ValueError(f"tier {self.tier!r} requires channels {sorted(expected)}")
        object.__setattr__(self, "channels", MappingProxyType(dict(self.channels)))


OBSERVATION_TIERS = MappingProxyType({
    "limited": MappingProxyType({
        "T": ChannelSpec(5, 2, 31, 0.05),
        "crack": ChannelSpec(3, 2, 11, 0.10),
    }),
    "intermediate": MappingProxyType({
        "T": ChannelSpec(9, 3, 61, 0.05),
        "pH2O": ChannelSpec(7, 2, 51, 0.10),
        "carbon": ChannelSpec(7, 3, 31, 0.08),
        "crack": ChannelSpec(7, 3, 31, 0.08),
        "stress": ChannelSpec(7, 2, 51, 0.10),
    }),
    "sufficient": MappingProxyType({
        "T": ChannelSpec(13, 4, 91, 0.03),
        "pO2": ChannelSpec(11, 3, 76, 0.07),
        "pH2O": ChannelSpec(11, 3, 76, 0.07),
        "pProducts": ChannelSpec(11, 3, 76, 0.07),
        "carbon": ChannelSpec(11, 4, 51, 0.05),
        "crack": ChannelSpec(11, 4, 51, 0.05),
        "stress": ChannelSpec(11, 3, 76, 0.07),
    }),
})


def sample_nozzle_observations(
    trajectory: NozzleFieldTrajectory,
    *,
    tier: str,
    noise_level: str,
    seed: int,
    timestamp_jitter_fraction: float = 0.20,
) -> NozzleObservationPacket:
    """Sample independent passive sensor clocks without exposing clean fields."""

    if tier not in OBSERVATION_TIERS:
        raise ValueError(f"unknown observation tier {tier!r}")
    if noise_level not in NOISE_MULTIPLIERS:
        raise ValueError(f"unknown noise level {noise_level!r}")
    if not 0.0 <= timestamp_jitter_fraction < 0.5:
        raise ValueError("timestamp_jitter_fraction must lie in [0, 0.5)")
    rng = np.random.default_rng(seed)
    multiplier = NOISE_MULTIPLIERS[noise_level]
    channels = {}
    for name, spec in OBSERVATION_TIERS[tier].items():
        clean = trajectory.states[name]
        s_indices = _sensor_indices(clean.shape[1], spec.meridional_sensors)
        x_indices = _sensor_indices(clean.shape[2], spec.thickness_sensors)
        times = _irregular_times(
            trajectory.t, min(spec.time_samples, trajectory.t.size),
            timestamp_jitter_fraction, rng,
        )
        sampled = np.empty((times.size, s_indices.size, x_indices.size))
        for local_s, grid_s in enumerate(s_indices):
            for local_x, grid_x in enumerate(x_indices):
                sampled[:, local_s, local_x] = np.interp(
                    times, trajectory.t, clean[:, grid_s, grid_x]
                )
        relative, floor = CHANNEL_NOISE[name]
        sigma = multiplier * max(relative * float(np.ptp(clean)), floor)
        if sigma > 0.0:
            sampled += rng.normal(0.0, sigma, sampled.shape)
        sampled = _clip_channel(name, sampled)
        retained = rng.random(sampled.shape) >= spec.missing_fraction
        retained[[0, -1], :, :] = True
        values = np.where(retained, sampled, np.nan)
        channels[name] = NozzleObservationChannel(
            name=name,
            times=times,
            s_indices=s_indices,
            x_indices=x_indices,
            values=values,
            retained=retained,
            noise_sigma=float(sigma),
        )
    return NozzleObservationPacket(
        condition_id=trajectory.mission_id,
        tier=tier,
        noise_level=noise_level,
        channels=channels,
        seed=int(seed),
    )


def reconstruct_nozzle_channel(
    packet: NozzleObservationPacket,
    channel_name: str,
    trajectory: NozzleFieldTrajectory,
) -> tuple[np.ndarray, dict[str, object]]:
    """Interpolate one observed channel onto its truth grid for weak projection."""

    channel = packet.channels[channel_name]
    reconstructed, diagnostics = reconstruct_nozzle_channel_on_grid(
        channel,
        target_t=trajectory.t,
        target_s=trajectory.s,
        target_x=trajectory.x,
    )
    clean = trajectory.states[channel_name]
    diagnostics["field_nrmse"] = float(
        np.sqrt(np.mean((reconstructed - clean) ** 2))
        / max(float(np.ptp(clean)), 1e-12)
    )
    return reconstructed, diagnostics


def reconstruct_nozzle_channel_on_grid(
    channel: NozzleObservationChannel,
    *,
    target_t: np.ndarray,
    target_s: np.ndarray,
    target_x: np.ndarray,
    temporal_constraint: str | None = None,
    spatial_method: str = "linear",
    spatial_smoothing_multiplier: float = 1.0,
    source_s: np.ndarray | None = None,
    source_x: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reconstruct observations without requiring access to clean truth fields.

    ``temporal_constraint`` is an optional physics prior for irreversible state
    variables.  It applies isotonic regression at each sensor followed by a
    monotonic cubic interpolant; it never reads a clean trajectory.
    """

    target_t = np.asarray(target_t, dtype=float)
    target_s = np.asarray(target_s, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    source_s = target_s if source_s is None else np.asarray(source_s, dtype=float)
    source_x = target_x if source_x is None else np.asarray(source_x, dtype=float)
    if (
        source_s.ndim != 1
        or source_x.ndim != 1
        or np.max(channel.s_indices) >= source_s.size
        or np.max(channel.x_indices) >= source_x.size
    ):
        raise ValueError("source coordinates must contain every sensor index")
    if temporal_constraint not in {None, "nondecreasing", "nonincreasing"}:
        raise ValueError(
            "temporal_constraint must be None, 'nondecreasing', or 'nonincreasing'"
        )
    if spatial_method not in {
        "linear",
        "pchip_meridional",
        "noise_calibrated_spline_meridional",
    }:
        raise ValueError(
            "spatial_method must be 'linear', 'pchip_meridional', or "
            "'noise_calibrated_spline_meridional'"
        )
    if spatial_smoothing_multiplier < 0.0:
        raise ValueError("spatial_smoothing_multiplier must be nonnegative")
    coarse = np.empty(
        (target_t.size, channel.s_indices.size, channel.x_indices.size),
        dtype=float,
    )
    retained_counts = []
    for local_s in range(channel.s_indices.size):
        for local_x in range(channel.x_indices.size):
            retained = channel.retained[:, local_s, local_x]
            retained_counts.append(int(np.sum(retained)))
            observed_times = channel.times[retained]
            observed_values = channel.values[retained, local_s, local_x]
            if temporal_constraint is None:
                coarse[:, local_s, local_x] = np.interp(
                    target_t, observed_times, observed_values
                )
            else:
                direction = 1.0 if temporal_constraint == "nondecreasing" else -1.0
                constrained = direction * _isotonic_increasing(
                    direction * observed_values
                )
                coarse[:, local_s, local_x] = PchipInterpolator(
                    observed_times, constrained, extrapolate=True
                )(target_t)
    sensor_s = source_s[channel.s_indices]
    sensor_x = source_x[channel.x_indices]
    if spatial_method == "linear":
        interpolator = RegularGridInterpolator(
            (target_t, sensor_s, sensor_x),
            coarse,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        tt, ss, xx = np.meshgrid(
            target_t, target_s, target_x, indexing="ij"
        )
        reconstructed = interpolator(
            np.column_stack((tt.ravel(), ss.ravel(), xx.ravel()))
        ).reshape((target_t.size, target_s.size, target_x.size))
    elif spatial_method == "pchip_meridional":
        meridional = PchipInterpolator(
            sensor_s,
            coarse,
            axis=1,
            extrapolate=True,
        )(target_s)
        reconstructed = np.empty(
            (target_t.size, target_s.size, target_x.size), dtype=float
        )
        for time_index in range(target_t.size):
            for s_index in range(target_s.size):
                reconstructed[time_index, s_index] = np.interp(
                    target_x,
                    sensor_x,
                    meridional[time_index, s_index],
                )
    else:
        meridional = np.empty(
            (target_t.size, target_s.size, sensor_x.size), dtype=float
        )
        smoothing_budget = (
            spatial_smoothing_multiplier
            * sensor_s.size
            * channel.noise_sigma**2
        )
        spline_degree = min(3, sensor_s.size - 1)
        for time_index in range(target_t.size):
            for x_index in range(sensor_x.size):
                meridional[time_index, :, x_index] = UnivariateSpline(
                    sensor_s,
                    coarse[time_index, :, x_index],
                    k=spline_degree,
                    s=smoothing_budget,
                    ext="extrapolate",
                )(target_s)
        reconstructed = np.empty(
            (target_t.size, target_s.size, target_x.size), dtype=float
        )
        for time_index in range(target_t.size):
            for s_index in range(target_s.size):
                reconstructed[time_index, s_index] = np.interp(
                    target_x,
                    sensor_x,
                    meridional[time_index, s_index],
                )
    reconstructed = _clip_channel(channel.name, reconstructed)
    diagnostics = {
        "retained_fraction": float(np.mean(channel.retained)),
        "mean_retained_time_samples": float(np.mean(retained_counts)),
        "spatial_sensor_fraction": float(
            channel.s_indices.size * channel.x_indices.size
            / (target_s.size * target_x.size)
        ),
        "nominal_time_fraction": float(channel.times.size / target_t.size),
        "temporal_constraint": temporal_constraint,
        "spatial_method": spatial_method,
        "spatial_smoothing_multiplier": float(spatial_smoothing_multiplier),
    }
    return reconstructed, diagnostics


def load_nozzle_observation_packet(
    path: str | Path,
    *,
    condition_id: str,
    tier: str,
    noise_level: str,
    seed: int,
) -> NozzleObservationPacket:
    """Load one observation-only NPZ packet produced by the R3 protocol."""

    with np.load(Path(path), allow_pickle=False) as data:
        channels = {}
        for name in OBSERVATION_TIERS[tier]:
            channels[name] = NozzleObservationChannel(
                name=name,
                times=data[f"times__{name}"],
                s_indices=data[f"s_indices__{name}"],
                x_indices=data[f"x_indices__{name}"],
                values=data[f"values__{name}"],
                retained=data[f"retained__{name}"],
                noise_sigma=float(data[f"noise_sigma__{name}"]),
            )
    return NozzleObservationPacket(
        condition_id=condition_id,
        tier=tier,
        noise_level=noise_level,
        channels=channels,
        seed=seed,
    )


def observation_packet_summary(packet: NozzleObservationPacket) -> dict[str, object]:
    return {
        "condition_id": packet.condition_id,
        "tier": packet.tier,
        "noise_level": packet.noise_level,
        "seed": packet.seed,
        "channels": {
            name: {
                "time_samples": int(channel.times.size),
                "meridional_sensors": int(channel.s_indices.size),
                "thickness_sensors": int(channel.x_indices.size),
                "retained_fraction": float(np.mean(channel.retained)),
                "noise_sigma": channel.noise_sigma,
            }
            for name, channel in packet.channels.items()
        },
    }


def _sensor_indices(size: int, requested: int) -> np.ndarray:
    count = min(size, requested)
    return np.unique(np.rint(np.linspace(0, size - 1, count)).astype(int))


def _irregular_times(t, count, jitter_fraction, rng):
    times = np.linspace(float(t[0]), float(t[-1]), count)
    if count > 2:
        spacing = (t[-1] - t[0]) / (count - 1)
        times[1:-1] += rng.uniform(
            -jitter_fraction * spacing, jitter_fraction * spacing, count - 2
        )
        times[1:-1] = np.sort(times[1:-1])
    return times


def _clip_channel(name: str, values: np.ndarray) -> np.ndarray:
    if name == "T":
        return np.maximum(values, 1.0)
    if name in {"pO2", "pH2O", "pProducts", "stress"}:
        return np.maximum(values, 0.0)
    if name == "carbon":
        return np.clip(values, 0.02, 1.0)
    if name == "crack":
        return np.clip(values, 0.0, 0.92)
    return values


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Unweighted pool-adjacent-violators fit for one sensor history."""

    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or observations.size < 2:
        raise ValueError("isotonic input must be a one-dimensional history")
    levels: list[float] = []
    weights: list[int] = []
    for value in observations:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (
                weights[-2] * levels[-2] + weights[-1] * levels[-1]
            ) / weight
            levels[-2:] = [level]
            weights[-2:] = [weight]
    return np.repeat(np.asarray(levels), np.asarray(weights, dtype=int))


def _readonly(values, dtype=float):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array
