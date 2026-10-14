"""Engineering-facing passive observation operators for nozzle truth fields.

The existing discovery benchmarks expose every latent field at multiple wall
depths.  This module instead projects a truth trajectory into modalities that
could be obtained in a ground-test programme: outer-wall infrared temperature,
strain-inferred stress, exit spectroscopy proxies, post-test NDE maps, and an
optional heavily instrumented research-rig tier with embedded thermocouples and
gas micro-sampling ports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .truth.reusable_nozzle import NozzleFieldTrajectory


TIERS = (
    "standard_ground_test",
    "instrumented_research_rig",
    "instrumented_rig_with_acoustic_emission",
)


@dataclass(frozen=True)
class EngineeringObservationVector:
    values: np.ndarray
    noise_std: np.ndarray
    labels: tuple[str, ...]
    blocks: tuple[str, ...]
    tier: str

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        noise = np.asarray(self.noise_std, dtype=float)
        if values.ndim != 1 or noise.shape != values.shape:
            raise ValueError("engineering observation values and noise must be vectors")
        if len(self.labels) != values.size or len(self.blocks) != values.size:
            raise ValueError("labels and blocks must match the observation vector")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(noise)):
            raise ValueError("engineering observations must be finite")
        if np.any(noise <= 0.0):
            raise ValueError("engineering observation noise must be positive")
        if self.tier not in TIERS:
            raise ValueError(f"unknown engineering observation tier {self.tier!r}")
        values = np.array(values, copy=True)
        noise = np.array(noise, copy=True)
        values.setflags(write=False)
        noise.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "noise_std", noise)


def project_engineering_observations(
    trajectory: NozzleFieldTrajectory,
    *,
    tier: str,
    time_samples: int = 12,
) -> EngineeringObservationVector:
    """Project one truth trajectory into a deterministic measurement vector."""

    if tier not in TIERS:
        raise ValueError(f"unknown engineering observation tier {tier!r}")
    if time_samples < 3:
        raise ValueError("at least three time samples are required")
    time_indices = _indices(trajectory.t.size, min(time_samples, trajectory.t.size))
    standard_s = _indices(trajectory.s.size, min(7, trajectory.s.size))
    stress_s = _indices(trajectory.s.size, min(5, trajectory.s.size))

    values: list[float] = []
    noise: list[float] = []
    labels: list[str] = []
    blocks: list[str] = []

    def append_block(name, block_values, sigma, block_labels):
        array = np.asarray(block_values, dtype=float).ravel()
        sigma_array = np.broadcast_to(np.asarray(sigma, dtype=float), array.shape)
        values.extend(array.tolist())
        noise.extend(sigma_array.tolist())
        labels.extend(block_labels)
        blocks.extend([name] * array.size)

    # External IR camera: outer surface only.
    ir = trajectory.states["T"][np.ix_(time_indices, standard_s, [-1])][:, :, 0]
    append_block(
        "outer_ir_temperature",
        ir,
        5.0,
        _time_space_labels("IR_T", trajectory, time_indices, standard_s),
    )

    # Strain gauges are represented as equivalent-stress readouts after a
    # separately calibrated elastic conversion.
    stress = trajectory.states["stress"][np.ix_(time_indices, stress_s, [-1])][:, :, 0]
    append_block(
        "outer_strain_inferred_stress",
        stress,
        2.0e6,
        _time_space_labels("strain_stress", trajectory, time_indices, stress_s),
    )

    # Exit-plane optical/emission spectroscopy proxy.  It is intentionally a
    # boundary signal, not an impossible measurement of every pore pressure.
    exit_index = trajectory.s.size - 1
    for species, relative, floor in (
        ("pO2", 0.03, 50.0),
        ("pH2O", 0.03, 10_000.0),
        ("pProducts", 0.05, 50.0),
    ):
        signal = trajectory.states[species][time_indices, exit_index, 0]
        sigma = max(relative * float(np.ptp(trajectory.states[species])), floor)
        append_block(
            f"exit_spectroscopy_{species}",
            signal,
            sigma,
            tuple(
                f"exit_{species}:t={trajectory.t[index]:.6g}"
                for index in time_indices
            ),
        )

    # Post-test NDE: a wall-thickness projection, not time-resolved latent data.
    crack_map = np.max(trajectory.states["crack"][-1, standard_s, :], axis=-1)
    append_block(
        "posttest_nde_crack",
        crack_map,
        0.002,
        tuple(f"NDE_crack:s={trajectory.s[index]:.6g}" for index in standard_s),
    )
    carbon_map = np.mean(trajectory.states["carbon"][-1, standard_s, :], axis=-1)
    append_block(
        "posttest_ct_density_carbon_proxy",
        carbon_map,
        0.003,
        tuple(f"CT_carbon:s={trajectory.s[index]:.6g}" for index in standard_s),
    )
    equivalent_mass_loss = np.asarray([
        np.mean(1.0 - trajectory.states["carbon"][-1])
    ])
    append_block(
        "posttest_equivalent_mass_loss",
        equivalent_mass_loss,
        5.0e-4,
        ("equivalent_mass_loss",),
    )

    if tier in {
        "instrumented_research_rig",
        "instrumented_rig_with_acoustic_emission",
    }:
        rig_s = _indices(trajectory.s.size, min(5, trajectory.s.size))
        tc_x = np.unique(
            np.asarray([1, max(1, trajectory.x.size // 2)], dtype=int).clip(
                0, trajectory.x.size - 1
            )
        )
        thermocouples = trajectory.states["T"][
            np.ix_(time_indices, rig_s, tc_x)
        ]
        append_block(
            "embedded_thermocouples",
            thermocouples,
            3.0,
            tuple(
                f"TC:t={trajectory.t[t]:.6g}:s={trajectory.s[s]:.6g}:x={trajectory.x[x]:.6g}"
                for t in time_indices for s in rig_s for x in tc_x
            ),
        )

        # Sacrificial micro-sampling ports on the hot-gas-side surface.  This is
        # a ground research rig, not an operational flight sensor assumption.
        for species, relative, floor in (
            ("pO2", 0.02, 30.0),
            ("pH2O", 0.02, 5_000.0),
            ("pProducts", 0.03, 30.0),
        ):
            signal = trajectory.states[species][np.ix_(time_indices, rig_s, [0])][
                :, :, 0
            ]
            sigma = max(relative * float(np.ptp(trajectory.states[species])), floor)
            append_block(
                f"inner_surface_microports_{species}",
                signal,
                sigma,
                _time_space_labels(
                    f"microport_{species}", trajectory, time_indices, rig_s
                ),
            )

        rig_x = np.unique(np.asarray([0, trajectory.x.size - 1], dtype=int))
        rig_stress = trajectory.states["stress"][
            np.ix_(time_indices, rig_s, rig_x)
        ]
        append_block(
            "dual_surface_strain_inferred_stress",
            rig_stress,
            2.0e6,
            tuple(
                f"dual_strain:t={trajectory.t[t]:.6g}:s={trajectory.s[s]:.6g}:x={trajectory.x[x]:.6g}"
                for t in time_indices for s in rig_s for x in rig_x
            ),
        )

        if tier == "instrumented_rig_with_acoustic_emission":
            # Distributed acoustic emission is represented by the positive
            # wall-averaged crack-rate proxy.  The mapping from event energy to
            # crack increment would require separate experimental calibration.
            crack_mean = np.mean(
                trajectory.states["crack"][:, rig_s, :], axis=2
            )
            crack_rate = np.maximum(
                np.gradient(crack_mean, trajectory.t, axis=0, edge_order=2),
                0.0,
            )
            sampled_rate = crack_rate[time_indices]
            sigma = max(0.05 * float(np.ptp(crack_rate)), 1.0e-8)
            append_block(
                "distributed_acoustic_crack_rate_proxy",
                sampled_rate,
                sigma,
                _time_space_labels(
                    "acoustic_crack_rate", trajectory, time_indices, rig_s
                ),
            )

    return EngineeringObservationVector(
        values=np.asarray(values),
        noise_std=np.asarray(noise),
        labels=tuple(labels),
        blocks=tuple(blocks),
        tier=tier,
    )


def sample_engineering_observations(
    trajectory: NozzleFieldTrajectory,
    *,
    tier: str,
    seed: int,
    time_samples: int = 12,
    missing_fraction: float = 0.0,
) -> tuple[EngineeringObservationVector, np.ndarray]:
    """Add declared measurement noise and return a retained-value mask."""

    if not 0.0 <= missing_fraction < 1.0:
        raise ValueError("missing_fraction must lie in [0, 1)")
    clean = project_engineering_observations(
        trajectory,
        tier=tier,
        time_samples=time_samples,
    )
    rng = np.random.default_rng(seed)
    observed = clean.values + rng.normal(0.0, clean.noise_std)
    retained = rng.random(clean.values.size) >= missing_fraction
    packet = EngineeringObservationVector(
        values=observed,
        noise_std=clean.noise_std,
        labels=clean.labels,
        blocks=clean.blocks,
        tier=clean.tier,
    )
    retained.setflags(write=False)
    return packet, retained


def _indices(size: int, count: int) -> np.ndarray:
    return np.unique(np.rint(np.linspace(0, size - 1, count)).astype(int))


def _time_space_labels(prefix, trajectory, time_indices, space_indices):
    return tuple(
        f"{prefix}:t={trajectory.t[t]:.6g}:s={trajectory.s[s]:.6g}"
        for t in time_indices for s in space_indices
    )
