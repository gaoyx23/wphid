"""Higher-fidelity thermo-chemo-mechanical C/SiC hot-structure truth source.

This model extends :mod:`wphid.truth.reusable_tps` with the minimum additional
physics required by NASA C/SiC/RCC oxidation studies: separate oxidant and
product-gas pressure fields, porous-medium pressure-driven transport, remaining
carbon, matrix/coating crack damage, and a thermo-mechanical stress field.

It remains a component-scale research model.  It does not replace trajectory
CFD, a full tensor finite-element stress analysis, or material qualification.
Its purpose is to generate substantially more credible ground truth for
mechanism-identification experiments than manufactured polynomial fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .reusable_tps import (
    TPSFieldTrajectory,
    TPSGrid,
    TPSMaterial,
    TPSMission,
    _conservative_diffusion,
    _external_oxidant,
    _heat_capacity,
    _initial_damage_field,
    _surface_heat_flux,
    _validate_configuration,
)


_STEFAN_BOLTZMANN = 5.670374419e-8
_GAS_CONSTANT = 8.314462618


@dataclass(frozen=True)
class CMCTransport:
    """Effective gas transport and crack-opening closure."""

    oxygen_diffusivity_closed: float = 3.0e-7
    product_diffusivity_closed: float = 3.8e-7
    open_to_closed_diffusivity_ratio: float = 18.0
    intrinsic_permeability: float = 4.0e-16
    gas_viscosity: float = 4.0e-5
    surface_mass_transfer: float = 2.5e-4
    reference_crack_opening: float = 1.0
    stress_opening_gain: float = 1.6
    damage_opening_gain: float = 5.0
    carbon_loss_opening_gain: float = 3.0
    high_temperature_closure: float = 0.45
    closure_temperature: float = 1500.0
    closure_temperature_span: float = 450.0


@dataclass(frozen=True)
class CMCOxidation:
    """Carbon reaction and gas source/sink closure."""

    activation_energy: float = 123_000.0
    pre_exponential: float = 0.11
    reference_oxygen_pressure: float = 20_000.0
    oxygen_exponent: float = 1.0
    carbon_exponent: float = 1.0
    oxygen_pressure_consumption: float = 4.0e5
    product_pressure_yield: float = 4.8e5
    reaction_heat_per_carbon_fraction: float = 2.2e8


@dataclass(frozen=True)
class CMCMechanics:
    """Reduced thermoelastic stress and matrix-cracking closure."""

    young_modulus: float = 60.0e9
    poisson_ratio: float = 0.032
    thermal_expansion_cold: float = 0.5e-6
    thermal_expansion_hot: float = 2.5e-6
    reference_temperature: float = 330.0
    restraint_fraction: float = 0.28
    peak_applied_stress: float = 42.0e6
    crack_threshold_stress: float = 38.0e6
    crack_rate: float = 2.2e-5
    crack_stress_exponent: float = 2.0
    oxidation_assisted_crack_rate: float = 1.5e-5
    initial_crack_damage: float = 0.015


def simulate_reusable_cmc_mission(
    mission_id: str,
    *,
    mission: TPSMission | None = None,
    material: TPSMaterial | None = None,
    transport: CMCTransport | None = None,
    oxidation: CMCOxidation | None = None,
    mechanics: CMCMechanics | None = None,
    grid: TPSGrid | None = None,
    initial_carbon_fraction: float | np.ndarray = 1.0,
    initial_crack_damage: float | np.ndarray | None = None,
) -> TPSFieldTrajectory:
    """Integrate heat, two gas species, carbon, cracks, and stress."""

    if not isinstance(mission_id, str) or not mission_id:
        raise ValueError("mission_id must be a non-empty string")
    mission = TPSMission() if mission is None else mission
    material = TPSMaterial() if material is None else material
    transport = CMCTransport() if transport is None else transport
    oxidation = CMCOxidation() if oxidation is None else oxidation
    mechanics = CMCMechanics() if mechanics is None else mechanics
    grid = TPSGrid() if grid is None else grid
    # Reuse the thermal/grid checks with a compatible three-field kinetics shell.
    from .reusable_tps import TPSOxidation

    _validate_configuration(mission, material, TPSOxidation(), grid)
    _validate_cmc_configuration(transport, oxidation, mechanics)

    dz = grid.axial_length / grid.axial_cells
    dx = grid.thickness / grid.thickness_cells
    z = (np.arange(grid.axial_cells, dtype=float) + 0.5) * dz
    x = (np.arange(grid.thickness_cells, dtype=float) + 0.5) * dx
    shape = (grid.axial_cells, grid.thickness_cells)
    steps = int(round(mission.duration / grid.time_step))
    output_stride = int(round(grid.output_interval / grid.time_step))
    if not np.isclose(steps * grid.time_step, mission.duration):
        raise ValueError("duration must be an integer multiple of time_step")
    if not np.isclose(output_stride * grid.time_step, grid.output_interval):
        raise ValueError("output_interval must be an integer multiple of time_step")

    temperature = np.full(shape, mission.support_temperature, dtype=float)
    oxygen_pressure = np.zeros(shape, dtype=float)
    product_pressure = np.zeros(shape, dtype=float)
    carbon_fraction = _fraction_field(initial_carbon_fraction, shape, "initial_carbon_fraction")
    crack_initial = (
        mechanics.initial_crack_damage
        if initial_crack_damage is None
        else initial_crack_damage
    )
    crack_damage = _initial_damage_field(crack_initial, shape)
    stress = np.zeros(shape, dtype=float)

    output_steps = np.arange(0, steps + 1, output_stride, dtype=int)
    if output_steps[-1] != steps:
        output_steps = np.append(output_steps, steps)
    times = output_steps.astype(float) * grid.time_step
    names = ("T", "pO2", "pCOx", "carbon", "crack", "stress")
    history = {name: np.empty((times.size, *shape), dtype=float) for name in names}
    heat_history = np.empty((times.size, grid.axial_cells), dtype=float)
    oxygen_boundary_history = np.empty(times.size, dtype=float)
    mechanical_load_history = np.empty((times.size, grid.axial_cells), dtype=float)

    output_index = 0
    for step in range(steps + 1):
        time_value = step * grid.time_step
        stress = _equivalent_stress(
            temperature,
            carbon_fraction,
            crack_damage,
            time_value,
            z,
            mission,
            grid,
            mechanics,
        )
        if step == output_steps[output_index]:
            history["T"][output_index] = temperature
            history["pO2"][output_index] = oxygen_pressure
            history["pCOx"][output_index] = product_pressure
            history["carbon"][output_index] = carbon_fraction
            history["crack"][output_index] = crack_damage
            history["stress"][output_index] = stress
            heat_history[output_index] = _surface_heat_flux(time_value, z, mission, grid)
            oxygen_boundary_history[output_index] = (
                oxidation.reference_oxygen_pressure * _external_oxidant(time_value, mission)
            )
            mechanical_load_history[output_index] = _mechanical_load(
                time_value, z, mission, grid, mechanics
            )
            output_index += 1
            if step == steps:
                break

        reaction = carbon_oxidation_rate(
            temperature, oxygen_pressure, carbon_fraction, oxidation
        )
        crack_opening = _relative_crack_opening(
            temperature,
            stress,
            carbon_fraction,
            crack_damage,
            transport,
            mechanics,
        )
        opening_fraction = np.clip(
            (crack_opening - transport.reference_crack_opening)
            / max(
                transport.stress_opening_gain
                + transport.damage_opening_gain
                + transport.carbon_loss_opening_gain,
                1e-12,
            ),
            0.0,
            1.0,
        )
        oxygen_diffusivity = transport.oxygen_diffusivity_closed * (
            1.0
            + (transport.open_to_closed_diffusivity_ratio - 1.0) * opening_fraction
        )
        product_diffusivity = transport.product_diffusivity_closed * (
            1.0
            + (transport.open_to_closed_diffusivity_ratio - 1.0) * opening_fraction
        )
        permeability = transport.intrinsic_permeability * crack_opening**2

        heat_capacity = _heat_capacity(temperature, material)
        stiffness_fraction = np.clip(
            carbon_fraction * (1.0 - crack_damage), 0.10, 1.0
        )
        conductivity_scale = np.clip(
            0.55 + 0.45 * stiffness_fraction,
            0.35,
            1.0,
        )
        heat_divergence = _conservative_diffusion(
            temperature,
            material.conductivity_axial * conductivity_scale,
            material.conductivity_through * conductivity_scale,
            dz,
            dx,
        )
        surface_temperature = temperature[:, 0]
        incident_heat = _surface_heat_flux(time_value, z, mission, grid)
        radiation = material.emissivity * _STEFAN_BOLTZMANN * (
            surface_temperature**4 - mission.ambient_temperature**4
        )
        heat_divergence[:, 0] += (incident_heat - radiation) / dx
        heat_divergence[:, -1] -= (
            mission.back_heat_transfer
            * (temperature[:, -1] - mission.support_temperature)
            / dx
        )
        temperature_tendency = (
            heat_divergence
            + oxidation.reaction_heat_per_carbon_fraction * reaction
        ) / (material.density * heat_capacity)

        total_pressure = oxygen_pressure + product_pressure
        oxygen_tendency = _gas_transport_tendency(
            oxygen_pressure,
            total_pressure,
            oxygen_diffusivity,
            permeability,
            transport.gas_viscosity,
            dz,
            dx,
        )
        product_tendency = _gas_transport_tendency(
            product_pressure,
            total_pressure,
            product_diffusivity,
            permeability,
            transport.gas_viscosity,
            dz,
            dx,
        )
        external_oxygen = (
            oxidation.reference_oxygen_pressure * _external_oxidant(time_value, mission)
        )
        oxygen_tendency[:, 0] += (
            transport.surface_mass_transfer
            * (external_oxygen - oxygen_pressure[:, 0])
            / dx
        )
        product_tendency[:, 0] += (
            transport.surface_mass_transfer * (0.0 - product_pressure[:, 0]) / dx
        )
        oxygen_tendency -= oxidation.oxygen_pressure_consumption * reaction
        product_tendency += oxidation.product_pressure_yield * reaction

        stress_ratio = np.maximum(
            stress / mechanics.crack_threshold_stress - 1.0, 0.0
        )
        crack_tendency = (
            mechanics.crack_rate
            * stress_ratio**mechanics.crack_stress_exponent
            * (1.0 - crack_damage)
            + mechanics.oxidation_assisted_crack_rate
            * (1.0 - carbon_fraction)
            * stress
            / mechanics.crack_threshold_stress
        )

        temperature = temperature + grid.time_step * temperature_tendency
        oxygen_pressure = np.clip(
            oxygen_pressure + grid.time_step * oxygen_tendency, 0.0, 1.5e5
        )
        product_pressure = np.clip(
            product_pressure + grid.time_step * product_tendency, 0.0, 1.5e5
        )
        carbon_fraction = np.clip(
            carbon_fraction - grid.time_step * reaction, 0.02, 1.0
        )
        crack_damage = np.clip(
            crack_damage + grid.time_step * crack_tendency, 0.0, 0.92
        )

        if not all(
            np.all(np.isfinite(values))
            for values in (
                temperature,
                oxygen_pressure,
                product_pressure,
                carbon_fraction,
                crack_damage,
            )
        ):
            raise RuntimeError("CMC integration produced a non-finite state")

    return TPSFieldTrajectory(
        t=times,
        z=z,
        x=x,
        states=history,
        loads={
            "surface_heat_flux": heat_history,
            "external_oxygen_pressure": oxygen_boundary_history,
            "mechanical_stress_load": mechanical_load_history,
        },
        mission_id=mission_id,
    )


def carbon_oxidation_rate(
    temperature: np.ndarray,
    oxygen_pressure: np.ndarray,
    carbon_fraction: np.ndarray,
    parameters: CMCOxidation | None = None,
) -> np.ndarray:
    """Arrhenius carbon-consumption rate with local oxygen availability."""

    parameters = CMCOxidation() if parameters is None else parameters
    oxygen_activity = np.maximum(oxygen_pressure, 0.0) / parameters.reference_oxygen_pressure
    return (
        parameters.pre_exponential
        * np.exp(
            -parameters.activation_energy
            / (_GAS_CONSTANT * np.maximum(temperature, 1.0))
        )
        * oxygen_activity**parameters.oxygen_exponent
        * np.clip(carbon_fraction, 0.0, 1.0) ** parameters.carbon_exponent
    )


def reusable_cmc_evaluation_truth() -> dict[str, object]:
    """Hidden ground-truth graph for post-identification scoring."""

    return {
        "fields": ("T", "pO2", "pCOx", "carbon", "crack", "stress"),
        "primary_hidden_mechanisms": {
            "carbon_oxidation": (
                "A*exp(-Ea/(R*T))*(pO2/p_ref)^m*carbon^q"
            ),
            "stress_assisted_cracking": (
                "k_c*positive(stress/sigma_c-1)^n*(1-crack)"
            ),
            "oxidation_assisted_cracking": (
                "k_ox*(1-carbon)*(stress/sigma_c)"
            ),
            "crack_controlled_transport": (
                "D_i(crack,carbon,stress,T), K(crack_opening)^2"
            ),
        },
        "topology": {
            "carbon_oxidation": ("T", "pO2", "pCOx", "carbon"),
            "stress_assisted_cracking": ("crack",),
            "oxidation_assisted_cracking": ("crack",),
            "crack_controlled_transport": ("pO2", "pCOx"),
            "property_degradation": ("T", "stress"),
        },
        "transport": asdict(CMCTransport()),
        "oxidation": asdict(CMCOxidation()),
        "mechanics": asdict(CMCMechanics()),
    }


def _equivalent_stress(
    temperature: np.ndarray,
    carbon_fraction: np.ndarray,
    crack_damage: np.ndarray,
    time_value: float,
    z: np.ndarray,
    mission: TPSMission,
    grid: TPSGrid,
    mechanics: CMCMechanics,
) -> np.ndarray:
    hot_fraction = np.clip(
        (temperature - 293.15) / (1873.15 - 293.15), 0.0, 1.0
    )
    expansion = mechanics.thermal_expansion_cold + hot_fraction * (
        mechanics.thermal_expansion_hot - mechanics.thermal_expansion_cold
    )
    mean_temperature = np.mean(temperature, axis=1, keepdims=True)
    thermal_mismatch = np.abs(temperature - mean_temperature) + (
        mechanics.restraint_fraction
        * np.maximum(mean_temperature - mechanics.reference_temperature, 0.0)
    )
    effective_modulus = mechanics.young_modulus * np.clip(
        carbon_fraction * (1.0 - crack_damage), 0.08, 1.0
    )
    thermal_stress = (
        effective_modulus
        * expansion
        * thermal_mismatch
        / (1.0 - mechanics.poisson_ratio)
    )
    applied = _mechanical_load(time_value, z, mission, grid, mechanics)[:, None]
    depth = np.linspace(0.0, 1.0, temperature.shape[1], dtype=float)[None, :]
    return np.sqrt(thermal_stress**2 + (applied * np.exp(-1.4 * depth)) ** 2)


def _mechanical_load(
    time_value: float,
    z: np.ndarray,
    mission: TPSMission,
    grid: TPSGrid,
    mechanics: CMCMechanics,
) -> np.ndarray:
    temporal = np.exp(-0.5 * ((time_value - mission.peak_time) / (0.9 * mission.heat_width)) ** 2)
    normalized_z = z / grid.axial_length
    spatial = 0.68 + 0.32 * np.exp(
        -0.5 * ((normalized_z - mission.hotspot_location) / mission.hotspot_width) ** 2
    )
    return mechanics.peak_applied_stress * temporal * spatial


def _relative_crack_opening(
    temperature: np.ndarray,
    stress: np.ndarray,
    carbon_fraction: np.ndarray,
    crack_damage: np.ndarray,
    transport: CMCTransport,
    mechanics: CMCMechanics,
) -> np.ndarray:
    stress_opening = transport.stress_opening_gain * np.clip(
        stress / mechanics.crack_threshold_stress, 0.0, 3.0
    )
    damage_opening = transport.damage_opening_gain * crack_damage
    carbon_opening = transport.carbon_loss_opening_gain * (1.0 - carbon_fraction)
    thermal_closure = transport.high_temperature_closure * np.clip(
        (temperature - transport.closure_temperature)
        / transport.closure_temperature_span,
        0.0,
        1.0,
    )
    return np.clip(
        transport.reference_crack_opening
        + stress_opening
        + damage_opening
        + carbon_opening
        - thermal_closure,
        0.20,
        12.0,
    )


def _gas_transport_tendency(
    species_pressure: np.ndarray,
    total_pressure: np.ndarray,
    diffusivity: np.ndarray,
    permeability: np.ndarray,
    viscosity: float,
    dz: float,
    dx: float,
) -> np.ndarray:
    tendency = _conservative_diffusion(
        species_pressure, diffusivity, diffusivity, dz, dx
    )
    tendency += _pressure_advection(
        species_pressure, total_pressure, permeability, viscosity, axis=0, spacing=dz
    )
    tendency += _pressure_advection(
        species_pressure, total_pressure, permeability, viscosity, axis=1, spacing=dx
    )
    return tendency


def _pressure_advection(
    species: np.ndarray,
    total_pressure: np.ndarray,
    permeability: np.ndarray,
    viscosity: float,
    *,
    axis: int,
    spacing: float,
) -> np.ndarray:
    pressure_difference = np.diff(total_pressure, axis=axis)
    left_slice = [slice(None)] * species.ndim
    right_slice = [slice(None)] * species.ndim
    left_slice[axis] = slice(None, -1)
    right_slice[axis] = slice(1, None)
    left_slice = tuple(left_slice)
    right_slice = tuple(right_slice)
    face_permeability = 0.5 * (
        permeability[left_slice] + permeability[right_slice]
    )
    velocity = -face_permeability * pressure_difference / (viscosity * spacing)
    upwind = np.where(velocity >= 0.0, species[left_slice], species[right_slice])
    flux = velocity * upwind
    tendency = np.zeros_like(species)
    tendency[left_slice] -= flux / spacing
    tendency[right_slice] += flux / spacing
    return tendency


def _fraction_field(
    value: float | np.ndarray, shape: tuple[int, int], name: str
) -> np.ndarray:
    field = np.asarray(value, dtype=float)
    if field.ndim == 0:
        field = np.full(shape, float(field), dtype=float)
    if field.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    if not np.all(np.isfinite(field)) or np.any((field <= 0.0) | (field > 1.0)):
        raise ValueError(f"{name} must be finite and lie in (0, 1]")
    return np.array(field, copy=True)


def _validate_cmc_configuration(
    transport: CMCTransport,
    oxidation: CMCOxidation,
    mechanics: CMCMechanics,
) -> None:
    values = [
        *asdict(transport).values(),
        *asdict(oxidation).values(),
        *asdict(mechanics).values(),
    ]
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("all CMC configuration values must be finite")
    if transport.oxygen_diffusivity_closed <= 0.0:
        raise ValueError("gas diffusivities must be positive")
    if transport.intrinsic_permeability < 0.0 or transport.gas_viscosity <= 0.0:
        raise ValueError("permeability must be non-negative and viscosity positive")
    if oxidation.reference_oxygen_pressure <= 0.0:
        raise ValueError("reference oxygen pressure must be positive")
    if mechanics.young_modulus <= 0.0 or mechanics.crack_threshold_stress <= 0.0:
        raise ValueError("mechanical stiffness and threshold must be positive")
