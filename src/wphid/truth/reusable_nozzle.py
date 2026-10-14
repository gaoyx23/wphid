"""Geometry-resolved C/SiC rocket-nozzle truth source.

The reference envelope is the publicly documented HM7-class C/SiC
demonstrator (0.988 m long, 0.938 m exit diameter and 24.7 kg).  The exact
manufacturer contour is not public, so a monotone cubic bell is constructed
from the documented envelope and HM7B expansion ratio.  Wall thickness is
inferred from the published mass, the bell surface area and the C/SiC density.

This is a mechanism-identification benchmark, not a flight-design model.  It
adds the geometry and propulsion-environment effects missing from the plate
baseline while reusing its material, cracking and weak-form conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..transport_mechanisms import TransportCandidate, modulation_fraction, constitutive_flux

from .reusable_cmc import CMCMechanics
from .csic_surface import (
    NozzleSurfaceChemistry,
    csic_surface_rates,
    validate_surface_chemistry,
)
from .reusable_tps import TPSMaterial, _heat_capacity, _readonly


_GAS_CONSTANT = 8.314462618
_STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass(frozen=True)
class NozzleGeometry:
    """Public HM7-class envelope and a reproducible smooth bell contour."""

    length: float = 0.988
    exit_radius: float = 0.469
    expansion_ratio: float = 62.5
    wall_thickness: float = 0.0065
    throat_angle_deg: float = 30.0
    exit_angle_deg: float = 7.0
    reference_mass: float = 24.7

    @property
    def throat_radius(self) -> float:
        return self.exit_radius / np.sqrt(self.expansion_ratio)

    def inner_radius(self, z: np.ndarray | float) -> np.ndarray:
        """Cubic-Hermite bell passing through throat and exit dimensions."""

        axial = np.asarray(z, dtype=float)
        xi = axial / self.length
        h00 = 2.0 * xi**3 - 3.0 * xi**2 + 1.0
        h10 = xi**3 - 2.0 * xi**2 + xi
        h01 = -2.0 * xi**3 + 3.0 * xi**2
        h11 = xi**3 - xi**2
        slope_0 = np.tan(np.deg2rad(self.throat_angle_deg))
        slope_1 = np.tan(np.deg2rad(self.exit_angle_deg))
        return (
            h00 * self.throat_radius
            + h10 * self.length * slope_0
            + h01 * self.exit_radius
            + h11 * self.length * slope_1
        )

    def inner_slope(self, z: np.ndarray | float) -> np.ndarray:
        axial = np.asarray(z, dtype=float)
        xi = axial / self.length
        slope_0 = np.tan(np.deg2rad(self.throat_angle_deg))
        slope_1 = np.tan(np.deg2rad(self.exit_angle_deg))
        return (
            (6.0 * xi**2 - 6.0 * xi) * self.throat_radius / self.length
            + (3.0 * xi**2 - 4.0 * xi + 1.0) * slope_0
            + (-6.0 * xi**2 + 6.0 * xi) * self.exit_radius / self.length
            + (3.0 * xi**2 - 2.0 * xi) * slope_1
        )

    def surface_area(self, points: int = 4097) -> float:
        z = np.linspace(0.0, self.length, points)
        radius = self.inner_radius(z)
        metric = np.sqrt(1.0 + self.inner_slope(z) ** 2)
        return float(2.0 * np.pi * np.trapezoid(radius * metric, z))

    def shell_mass(self, density: float = 1900.0) -> float:
        return self.surface_area() * self.wall_thickness * density


@dataclass(frozen=True)
class NozzleGrid:
    meridional_cells: int = 31
    thickness_cells: int = 8
    time_step: float = 0.02
    output_interval: float = 5.0


@dataclass(frozen=True)
class NozzleMission:
    """Natural hot-fire boundary conditions, without imposed state impulses."""

    duration: float = 900.0
    chamber_pressure: float = 3.6e6
    mixture_ratio: float = 5.04
    stagnation_temperature: float = 3500.0
    heat_flux_throat: float = 575_000.0
    steam_mole_fraction: float = 0.72
    residual_oxygen_fraction: float = 0.0002
    outer_oxygen_pressure: float = 21_000.0
    ramp_time: float = 8.0
    shutdown_time: float = 8.0
    initial_temperature: float = 330.0
    ambient_temperature: float = 300.0


@dataclass(frozen=True)
class NozzleThermalBoundary:
    """Axial gas-side heat-flux closure.

    ``legacy_radius_power`` preserves the frozen R2--R93 truth source.
    ``bartz_area_profile`` uses the classical Bartz area exponent together
    with an adiabatic-wall-temperature driving potential, normalized to the
    mission's declared throat heat flux.  It is a reduced engineering
    correlation, not a conjugate CFD boundary.
    """

    model: str = "legacy_radius_power"
    radius_exponent: float = 0.82
    bartz_area_exponent: float = 0.90
    gamma: float = 1.22
    prandtl: float = 0.70
    wall_reference_temperature: float = 330.0


@dataclass(frozen=True)
class NozzleTransport:
    oxygen_diffusivity_closed: float = 3.0e-7
    steam_diffusivity_closed: float = 4.2e-7
    product_diffusivity_closed: float = 3.8e-7
    open_to_closed_diffusivity_ratio: float = 18.0
    surface_mass_transfer: float = 2.5e-4
    stress_opening_gain: float = 1.6
    damage_opening_gain: float = 5.0
    carbon_loss_opening_gain: float = 3.0
    high_temperature_closure: float = 0.45
    closure_temperature: float = 1500.0
    closure_temperature_span: float = 450.0
    meridional_to_through_diffusivity_ratio: float = 1.0
    forchheimer_inverse_flux_scale: float = 0.0


@dataclass(frozen=True)
class NozzleChemistry:
    """Separate oxygen and steam attack paths for C/SiC."""

    oxygen_activation_energy: float = 123_000.0
    oxygen_pre_exponential: float = 0.11
    oxygen_reference_pressure: float = 20_000.0
    oxygen_exponent: float = 1.0
    steam_activation_energy: float = 145_000.0
    steam_pre_exponential: float = 0.42
    steam_reference_pressure: float = 1.0e6
    steam_exponent: float = 0.75
    carbon_exponent: float = 1.0
    oxygen_pressure_consumption: float = 4.0e5
    steam_pressure_consumption: float = 7.0e5
    product_pressure_yield: float = 5.2e5
    reaction_heat_per_carbon_fraction: float = 2.2e8


@dataclass(frozen=True)
class NozzleFieldTrajectory:
    t: np.ndarray
    s: np.ndarray
    x: np.ndarray
    z: np.ndarray
    r: np.ndarray
    states: Mapping[str, np.ndarray]
    loads: Mapping[str, np.ndarray]
    mission_id: str
    surface_states: Mapping[str, np.ndarray] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _readonly(self.t)
        meridional = _readonly(self.s)
        depth = _readonly(self.x)
        axial = _readonly(self.z)
        radius = _readonly(self.r)
        if time.ndim != 1 or meridional.ndim != 1 or depth.ndim != 1:
            raise ValueError("t, s and x must be one-dimensional")
        if axial.shape != (meridional.size, depth.size) or radius.shape != axial.shape:
            raise ValueError("z and r must map every shell cell")
        expected = (time.size, meridional.size, depth.size)
        states = {}
        for name, values in self.states.items():
            array = _readonly(values)
            if array.shape != expected:
                raise ValueError(f"state {name!r} must have shape {expected}")
            states[name] = array
        loads = {}
        for name, values in self.loads.items():
            array = _readonly(values)
            if array.shape[0] != time.size:
                raise ValueError(f"load {name!r} must match the time axis")
            loads[name] = array
        surface_states = {}
        for name, values in self.surface_states.items():
            array = _readonly(values)
            if array.shape != (time.size, meridional.size):
                raise ValueError(
                    f"surface state {name!r} must have shape "
                    f"{(time.size, meridional.size)}"
                )
            surface_states[name] = array
        object.__setattr__(self, "t", time)
        object.__setattr__(self, "s", meridional)
        object.__setattr__(self, "x", depth)
        object.__setattr__(self, "z", axial)
        object.__setattr__(self, "r", radius)
        object.__setattr__(self, "states", MappingProxyType(states))
        object.__setattr__(self, "loads", MappingProxyType(loads))
        object.__setattr__(self, "surface_states", MappingProxyType(surface_states))


def simulate_reusable_nozzle_mission(
    mission_id: str,
    *,
    mission: NozzleMission | None = None,
    geometry: NozzleGeometry | None = None,
    grid: NozzleGrid | None = None,
    material: TPSMaterial | None = None,
    thermal_boundary: NozzleThermalBoundary | None = None,
    surface_chemistry: NozzleSurfaceChemistry | None = None,
    transport: NozzleTransport | None = None,
    transport_candidate: TransportCandidate | None = None,
    chemistry: NozzleChemistry | None = None,
    mechanics: CMCMechanics | None = None,
    initial_carbon_fraction: float | np.ndarray = 1.0,
    initial_crack_damage: float | np.ndarray = 0.015,
    initial_states: Mapping[str, float | np.ndarray] | None = None,
    initial_surface_states: Mapping[str, float | np.ndarray] | None = None,
) -> NozzleFieldTrajectory:
    """Integrate seven coupled fields on the curved nozzle wall."""

    if not isinstance(mission_id, str) or not mission_id:
        raise ValueError("mission_id must be a non-empty string")
    mission = NozzleMission() if mission is None else mission
    geometry = NozzleGeometry() if geometry is None else geometry
    grid = NozzleGrid() if grid is None else grid
    material = TPSMaterial() if material is None else material
    thermal_boundary = (
        NozzleThermalBoundary() if thermal_boundary is None else thermal_boundary
    )
    transport = NozzleTransport() if transport is None else transport
    if transport_candidate is not None:
        if not isinstance(transport_candidate, TransportCandidate):
            raise TypeError("transport_candidate must be a TransportCandidate")
        transport = transport_candidate.configure_transport(transport)
    chemistry = NozzleChemistry() if chemistry is None else chemistry
    mechanics = CMCMechanics() if mechanics is None else mechanics
    _validate(
        geometry,
        grid,
        mission,
        material,
        thermal_boundary,
        transport,
        chemistry,
        mechanics,
    )
    if surface_chemistry is not None:
        validate_surface_chemistry(surface_chemistry)

    mesh = _shell_mesh(geometry, grid)
    shape = mesh["r"].shape
    steps = int(round(mission.duration / grid.time_step))
    stride = int(round(grid.output_interval / grid.time_step))
    if not np.isclose(steps * grid.time_step, mission.duration):
        raise ValueError("duration must be an integer multiple of time_step")
    if not np.isclose(stride * grid.time_step, grid.output_interval):
        raise ValueError("output_interval must be an integer multiple of time_step")

    initial_states = {} if initial_states is None else dict(initial_states)
    allowed_initial_states = {"T", "pO2", "pH2O", "pProducts", "carbon", "crack"}
    unknown_initial_states = set(initial_states).difference(allowed_initial_states)
    if unknown_initial_states:
        raise ValueError(
            f"unknown initial state fields: {', '.join(sorted(unknown_initial_states))}"
        )
    temperature = _initial_nozzle_field(
        initial_states.get("T", mission.initial_temperature),
        shape,
        name="initial temperature",
        lower=1.0,
        upper=5000.0,
    )
    oxygen = _initial_nozzle_field(
        initial_states.get("pO2", 0.0), shape, name="initial pO2",
        lower=0.0, upper=5.0e6,
    )
    steam = _initial_nozzle_field(
        initial_states.get("pH2O", 0.0), shape, name="initial pH2O",
        lower=0.0, upper=5.0e6,
    )
    products = _initial_nozzle_field(
        initial_states.get("pProducts", 0.0), shape, name="initial pProducts",
        lower=0.0, upper=5.0e6,
    )
    pressure_ceiling = max(
        1.5e5,
        1.2 * mission.chamber_pressure,
        1.2 * mission.outer_oxygen_pressure,
        float(np.max(oxygen)),
        float(np.max(steam)),
        float(np.max(products)),
    )
    carbon = _initial_nozzle_field(
        initial_states.get("carbon", initial_carbon_fraction),
        shape,
        name="initial_carbon_fraction",
        lower=0.02,
        upper=1.0,
    )
    crack = _initial_nozzle_field(
        initial_states.get("crack", initial_crack_damage),
        shape,
        name="initial_crack_damage",
        lower=0.0,
        upper=0.92,
    )
    stress = np.zeros(shape, dtype=float)
    surface_enabled = surface_chemistry is not None
    initial_surface_states = (
        {} if initial_surface_states is None else dict(initial_surface_states)
    )
    unknown_surface_states = set(initial_surface_states).difference(
        {"silica_thickness", "recession_depth"}
    )
    if unknown_surface_states:
        raise ValueError(
            "unknown initial surface states: "
            + ", ".join(sorted(unknown_surface_states))
        )
    if not surface_enabled and initial_surface_states:
        raise ValueError(
            "initial_surface_states require an enabled surface_chemistry model"
        )
    if surface_enabled:
        silica = _initial_surface_field(
            initial_surface_states.get(
                "silica_thickness",
                surface_chemistry.initial_silica_thickness,
            ),
            shape[0],
            name="initial silica thickness",
            lower=0.0,
            upper=surface_chemistry.maximum_silica_thickness,
        )
        recession = _initial_surface_field(
            initial_surface_states.get("recession_depth", 0.0),
            shape[0],
            name="initial recession depth",
            lower=0.0,
            upper=geometry.wall_thickness,
        )
    else:
        silica = recession = None

    output_steps = np.arange(0, steps + 1, stride, dtype=int)
    if output_steps[-1] != steps:
        output_steps = np.append(output_steps, steps)
    times = output_steps * grid.time_step
    names = ("T", "pO2", "pH2O", "pProducts", "carbon", "crack", "stress")
    history = {name: np.empty((times.size, *shape)) for name in names}
    load_history = {
        "surface_heat_flux": np.empty((times.size, shape[0])),
        "gas_pressure": np.empty((times.size, shape[0])),
        "gas_temperature": np.empty((times.size, shape[0])),
        "external_oxygen_pressure": np.empty((times.size, shape[0])),
        "internal_oxygen_pressure": np.empty((times.size, shape[0])),
        "external_steam_pressure": np.empty((times.size, shape[0])),
    }
    surface_history = {}
    if surface_enabled:
        surface_history = {
            "silica_thickness": np.empty((times.size, shape[0])),
            "recession_depth": np.empty((times.size, shape[0])),
        }
        for name in (
            "passive_formation_rate",
            "steam_volatilization_rate",
            "stress_spallation_rate",
            "active_recession_rate",
            "carbon_recession_rate",
            "total_recession_rate",
            "active_oxidation_fraction",
            "barrier_factor",
        ):
            load_history[f"surface_{name}"] = np.empty((times.size, shape[0]))

    flow = _quasi_one_dimensional_flow(mesh["z_inner"], geometry, mission)
    output_index = 0
    for step in range(steps + 1):
        time_value = step * grid.time_step
        duty = _hot_fire_duty(time_value, mission)
        heat_flux = duty * mission.heat_flux_throat * nozzle_heat_flux_shape(
            mesh["z_inner"],
            geometry=geometry,
            mission=mission,
            boundary=thermal_boundary,
        )
        gas_pressure = duty * flow["pressure"]
        external_steam = mission.steam_mole_fraction * gas_pressure
        internal_oxygen = mission.residual_oxygen_fraction * gas_pressure
        external_oxygen = np.full_like(gas_pressure, mission.outer_oxygen_pressure)
        stress = _nozzle_stress(
            temperature, carbon, crack, gas_pressure, mesh, geometry, mechanics
        )

        oxygen_rate, steam_rate = nozzle_reaction_rates(
            temperature, oxygen, steam, carbon, chemistry
        )
        if surface_enabled:
            coverage = np.clip(
                silica / surface_chemistry.maximum_silica_thickness,
                0.0,
                1.0,
            )
            barrier = np.exp(-surface_chemistry.silica_barrier_strength * coverage)
            oxygen_rate[:, 0] *= barrier
            steam_rate[:, 0] *= barrier
            surface_rates = csic_surface_rates(
                temperature[:, 0],
                oxygen[:, 0],
                steam[:, 0],
                stress[:, 0],
                silica,
                oxygen_rate[:, 0] + steam_rate[:, 0],
                surface_chemistry,
            )
        else:
            barrier = np.ones(shape[0], dtype=float)
            surface_rates = None

        if step == output_steps[output_index]:
            for name, values in (
                ("T", temperature), ("pO2", oxygen), ("pH2O", steam),
                ("pProducts", products), ("carbon", carbon),
                ("crack", crack), ("stress", stress),
            ):
                history[name][output_index] = values
            load_history["surface_heat_flux"][output_index] = heat_flux
            load_history["gas_pressure"][output_index] = gas_pressure
            load_history["gas_temperature"][output_index] = duty * flow["temperature"]
            load_history["external_oxygen_pressure"][output_index] = external_oxygen
            load_history["internal_oxygen_pressure"][output_index] = internal_oxygen
            load_history["external_steam_pressure"][output_index] = external_steam
            if surface_enabled:
                surface_history["silica_thickness"][output_index] = silica
                surface_history["recession_depth"][output_index] = recession
                for name in (
                    "passive_formation_rate",
                    "steam_volatilization_rate",
                    "stress_spallation_rate",
                    "active_recession_rate",
                    "carbon_recession_rate",
                    "total_recession_rate",
                    "active_oxidation_fraction",
                    "barrier_factor",
                ):
                    load_history[f"surface_{name}"][output_index] = surface_rates[name]
            output_index += 1
            if step == steps:
                break

        total_reaction = oxygen_rate + steam_rate
        opening = _crack_opening(temperature, stress, carbon, crack, transport, mechanics)
        opening_fraction = np.clip((opening - 1.0) / 9.6, 0.0, 1.0)
        if transport_candidate is not None:
            opening_fraction = modulation_fraction(
                transport_candidate.modulation, temperature=temperature,
                stress=stress, carbon=carbon, crack=crack,
                transport=transport, mechanics=mechanics,
            )
        diffusivity_gain = 1.0 + (
            transport.open_to_closed_diffusivity_ratio - 1.0
        ) * opening_fraction

        stiffness = np.clip(carbon * (1.0 - crack), 0.10, 1.0)
        conductivity_scale = np.clip(0.55 + 0.45 * stiffness, 0.35, 1.0)
        heat_divergence = _shell_diffusion(
            temperature,
            material.conductivity_axial * conductivity_scale,
            material.conductivity_through * conductivity_scale,
            mesh,
        )
        heat_divergence[:, 0] += heat_flux * mesh["inner_area"] / mesh["volume"][:, 0]
        outer_temperature = temperature[:, -1]
        radiation = material.emissivity * _STEFAN_BOLTZMANN * (
            outer_temperature**4 - mission.ambient_temperature**4
        )
        heat_divergence[:, -1] -= radiation * mesh["outer_area"] / mesh["volume"][:, -1]
        temperature_tendency = (
            heat_divergence + chemistry.reaction_heat_per_carbon_fraction * total_reaction
        ) / (material.density * _heat_capacity(temperature, material))

        oxygen_tendency = _shell_diffusion(
            oxygen,
            transport.meridional_to_through_diffusivity_ratio
            * transport.oxygen_diffusivity_closed
            * diffusivity_gain,
            transport.oxygen_diffusivity_closed * diffusivity_gain,
            mesh,
            forchheimer_inverse_flux_scale=(
                transport.forchheimer_inverse_flux_scale
            ),
        )
        steam_tendency = _shell_diffusion(
            steam,
            transport.meridional_to_through_diffusivity_ratio
            * transport.steam_diffusivity_closed
            * diffusivity_gain,
            transport.steam_diffusivity_closed * diffusivity_gain,
            mesh,
            forchheimer_inverse_flux_scale=(
                transport.forchheimer_inverse_flux_scale
            ),
        )
        product_tendency = _shell_diffusion(
            products,
            transport.meridional_to_through_diffusivity_ratio
            * transport.product_diffusivity_closed
            * diffusivity_gain,
            transport.product_diffusivity_closed * diffusivity_gain,
            mesh,
            forchheimer_inverse_flux_scale=(
                transport.forchheimer_inverse_flux_scale
            ),
        )
        boundary_scale = (
            transport.surface_mass_transfer
            * mesh["inner_area"]
            / mesh["volume"][:, 0]
            * barrier
        )
        oxygen_tendency[:, 0] += boundary_scale * (internal_oxygen - oxygen[:, 0])
        steam_tendency[:, 0] += boundary_scale * (external_steam - steam[:, 0])
        product_tendency[:, 0] += boundary_scale * (0.0 - products[:, 0])
        outer_boundary_scale = (
            transport.surface_mass_transfer
            * mesh["outer_area"] / mesh["volume"][:, -1]
        )
        oxygen_tendency[:, -1] += outer_boundary_scale * (
            external_oxygen - oxygen[:, -1]
        )
        oxygen_tendency -= chemistry.oxygen_pressure_consumption * oxygen_rate
        steam_tendency -= chemistry.steam_pressure_consumption * steam_rate
        product_tendency += chemistry.product_pressure_yield * total_reaction

        stress_ratio = np.maximum(stress / mechanics.crack_threshold_stress - 1.0, 0.0)
        crack_tendency = (
            mechanics.crack_rate
            * stress_ratio**mechanics.crack_stress_exponent
            * (1.0 - crack)
            + mechanics.oxidation_assisted_crack_rate
            * (1.0 - carbon)
            * stress / mechanics.crack_threshold_stress
        )

        dt = grid.time_step
        temperature += dt * temperature_tendency
        oxygen = np.clip(oxygen + dt * oxygen_tendency, 0.0, pressure_ceiling)
        steam = np.clip(steam + dt * steam_tendency, 0.0, pressure_ceiling)
        products = np.clip(products + dt * product_tendency, 0.0, pressure_ceiling)
        carbon = np.clip(carbon - dt * total_reaction, 0.02, 1.0)
        crack = np.clip(crack + dt * crack_tendency, 0.0, 0.92)
        if surface_enabled:
            silica = np.clip(
                silica + dt * surface_rates["silica_net_rate"],
                0.0,
                surface_chemistry.maximum_silica_thickness,
            )
            recession = np.maximum(
                recession + dt * surface_rates["total_recession_rate"],
                0.0,
            )
            if np.any(
                recession
                > surface_chemistry.fixed_domain_recession_fraction_limit
                * geometry.wall_thickness
            ):
                raise RuntimeError(
                    "surface recession exceeded the fixed-domain validity limit"
                )
        if not all(np.all(np.isfinite(v)) for v in (temperature, oxygen, steam, products, carbon, crack)):
            raise RuntimeError("nozzle integration produced a non-finite state")

    return NozzleFieldTrajectory(
        t=times,
        s=mesh["s"],
        x=mesh["x"],
        z=mesh["z"],
        r=mesh["r"],
        states=history,
        loads=load_history,
        mission_id=mission_id,
        surface_states=surface_history,
    )


def nozzle_reaction_rates(temperature, oxygen, steam, carbon, parameters=None):
    parameters = NozzleChemistry() if parameters is None else parameters
    safe_temperature = np.maximum(np.asarray(temperature, dtype=float), 1.0)
    remaining = np.clip(np.asarray(carbon, dtype=float), 0.0, 1.0)
    oxygen_activity = np.maximum(oxygen, 0.0) / parameters.oxygen_reference_pressure
    steam_activity = np.maximum(steam, 0.0) / parameters.steam_reference_pressure
    oxygen_rate = (
        parameters.oxygen_pre_exponential
        * np.exp(-parameters.oxygen_activation_energy / (_GAS_CONSTANT * safe_temperature))
        * oxygen_activity**parameters.oxygen_exponent
        * remaining**parameters.carbon_exponent
    )
    steam_rate = (
        parameters.steam_pre_exponential
        * np.exp(-parameters.steam_activation_energy / (_GAS_CONSTANT * safe_temperature))
        * steam_activity**parameters.steam_exponent
        * remaining**parameters.carbon_exponent
    )
    return oxygen_rate, steam_rate


def reusable_nozzle_evaluation_truth() -> dict[str, object]:
    return {
        "fields": ("T", "pO2", "pH2O", "pProducts", "carbon", "crack", "stress"),
        "primary_hidden_mechanisms": {
            "oxygen_carbon_oxidation": "A_o*exp(-Ea_o/(R*T))*pO2^m*carbon^q",
            "steam_corrosion": "A_w*exp(-Ea_w/(R*T))*pH2O^n*carbon^q",
            "stress_assisted_cracking": "k_c*positive(stress/sigma_c-1)^a*(1-crack)",
            "chemistry_assisted_cracking": "k_ox*(1-carbon)*(stress/sigma_c)",
            "crack_controlled_transport": "D_i(crack,carbon,stress,T)",
        },
        "topology": {
            "oxygen_carbon_oxidation": ("T", "pO2", "pProducts", "carbon"),
            "steam_corrosion": ("T", "pH2O", "pProducts", "carbon"),
            "stress_assisted_cracking": ("crack",),
            "chemistry_assisted_cracking": ("crack",),
            "crack_controlled_transport": ("pO2", "pH2O", "pProducts"),
            "property_degradation": ("T", "stress"),
        },
        "geometry": asdict(NozzleGeometry()),
        "mission": asdict(NozzleMission()),
        "chemistry": asdict(NozzleChemistry()),
    }


def reference_nozzle_missions() -> dict[str, NozzleMission]:
    """Frozen natural hot-fire conditions shared by qualification and discovery."""

    return {
        "lower_load_750s": NozzleMission(
            duration=750.0,
            chamber_pressure=3.2e6,
            heat_flux_throat=500_000.0,
            steam_mole_fraction=0.68,
        ),
        "nominal_900s": NozzleMission(),
        "high_thermal_750s": NozzleMission(
            duration=750.0,
            chamber_pressure=3.6e6,
            heat_flux_throat=650_000.0,
            steam_mole_fraction=0.72,
        ),
        "high_steam_900s": NozzleMission(
            duration=900.0,
            chamber_pressure=3.8e6,
            heat_flux_throat=610_000.0,
            steam_mole_fraction=0.82,
        ),
    }


def bartz_scaled_heat_flux(
    chamber_pressure: float,
    *,
    reference_pressure: float = 3.6e6,
    reference_heat_flux: float = 575_000.0,
) -> float:
    """Pressure-coupled throat heat flux using the simplified Bartz exponent."""

    values = (chamber_pressure, reference_pressure, reference_heat_flux)
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError("Bartz scaling inputs must be finite and positive")
    return float(reference_heat_flux * (chamber_pressure / reference_pressure) ** 0.8)


def nozzle_heat_flux_shape(
    z: np.ndarray,
    *,
    geometry: NozzleGeometry | None = None,
    mission: NozzleMission | None = None,
    boundary: NozzleThermalBoundary | None = None,
) -> np.ndarray:
    """Return the dimensionless axial heat-flux shape.

    Both supported closures equal one at the geometric throat.  The
    Bartz-type option retains only factors whose axial variation can be
    evaluated from this reduced model; unknown absolute gas-property factors
    are absorbed by ``mission.heat_flux_throat``.
    """

    geometry = NozzleGeometry() if geometry is None else geometry
    mission = NozzleMission() if mission is None else mission
    boundary = NozzleThermalBoundary() if boundary is None else boundary
    axial = np.asarray(z, dtype=float)
    radius = geometry.inner_radius(axial)
    if boundary.model == "legacy_radius_power":
        return (geometry.throat_radius / radius) ** boundary.radius_exponent
    if boundary.model != "bartz_area_profile":
        raise ValueError(f"unknown thermal boundary model {boundary.model!r}")

    area_ratio = (radius / geometry.throat_radius) ** 2
    mach = np.asarray([
        _supersonic_mach(float(value), boundary.gamma) for value in area_ratio
    ])
    compressibility = 1.0 + 0.5 * (boundary.gamma - 1.0) * mach**2
    static_temperature = mission.stagnation_temperature / compressibility
    recovery = boundary.prandtl ** (1.0 / 3.0)
    adiabatic_wall_temperature = static_temperature * (
        1.0 + recovery * 0.5 * (boundary.gamma - 1.0) * mach**2
    )
    driving = np.maximum(
        adiabatic_wall_temperature - boundary.wall_reference_temperature,
        1.0,
    )
    throat_compressibility = 1.0 + 0.5 * (boundary.gamma - 1.0)
    throat_static_temperature = (
        mission.stagnation_temperature / throat_compressibility
    )
    throat_adiabatic_wall_temperature = throat_static_temperature * (
        1.0 + recovery * 0.5 * (boundary.gamma - 1.0)
    )
    throat_driving = max(
        throat_adiabatic_wall_temperature
        - boundary.wall_reference_temperature,
        1.0,
    )
    return (
        area_ratio ** (-boundary.bartz_area_exponent)
        * driving
        / throat_driving
    )


def nozzle_shell_coordinates(
    geometry: NozzleGeometry | None = None,
    grid: NozzleGrid | None = None,
) -> dict[str, np.ndarray]:
    """Return observation-facing coordinates without exposing any truth state."""

    geometry = NozzleGeometry() if geometry is None else geometry
    grid = NozzleGrid() if grid is None else grid
    mesh = _shell_mesh(geometry, grid)
    coordinates = {
        name: np.array(mesh[name], copy=True)
        for name in ("s", "x", "z", "r")
    }
    for values in coordinates.values():
        values.setflags(write=False)
    return coordinates


def _shell_mesh(geometry: NozzleGeometry, grid: NozzleGrid) -> dict[str, np.ndarray]:
    z_face = np.linspace(0.0, geometry.length, grid.meridional_cells + 1)
    r_face = geometry.inner_radius(z_face)
    ds = np.sqrt(np.diff(z_face) ** 2 + np.diff(r_face) ** 2)
    s_face = np.concatenate(([0.0], np.cumsum(ds)))
    s = 0.5 * (s_face[:-1] + s_face[1:])
    z_inner = 0.5 * (z_face[:-1] + z_face[1:])
    r_inner = geometry.inner_radius(z_inner)
    slope = geometry.inner_slope(z_inner)
    normal_r = 1.0 / np.sqrt(1.0 + slope**2)
    normal_z = -slope * normal_r
    dx = geometry.wall_thickness / grid.thickness_cells
    x_face = np.linspace(0.0, geometry.wall_thickness, grid.thickness_cells + 1)
    x = 0.5 * (x_face[:-1] + x_face[1:])
    z = z_inner[:, None] + normal_z[:, None] * x[None, :]
    r = r_inner[:, None] + normal_r[:, None] * x[None, :]
    volume = r * ds[:, None] * dx
    inner_area = r_inner * ds
    outer_area = (r_inner + normal_r * geometry.wall_thickness) * ds
    slope_face = geometry.inner_slope(z_face)
    normal_r_face = 1.0 / np.sqrt(1.0 + slope_face**2)
    radius_s_face = r_face[:, None] + normal_r_face[:, None] * x[None, :]
    radius_x_face = r_inner[:, None] + normal_r[:, None] * x_face[None, :]
    return {
        "z_face": z_face, "r_face": r_face, "s_face": s_face,
        "s": s, "x": x, "z": z, "r": r, "ds": ds, "dx": dx,
        "z_inner": z_inner, "r_inner": r_inner, "volume": volume,
        "inner_area": inner_area, "outer_area": outer_area,
        "radius_s_face": radius_s_face, "radius_x_face": radius_x_face,
    }


def _shell_diffusion(
    field,
    coefficient_s,
    coefficient_x,
    mesh,
    *,
    forchheimer_inverse_flux_scale=0.0,
):
    tendency = np.zeros_like(field)
    ds, dx, volume = mesh["ds"], mesh["dx"], mesh["volume"]
    coefficient_s_face = 0.5 * (coefficient_s[1:] + coefficient_s[:-1])
    distance_s = 0.5 * (ds[1:] + ds[:-1])
    flux_s = -coefficient_s_face * (field[1:] - field[:-1]) / distance_s[:, None]
    if forchheimer_inverse_flux_scale > 0.0:
        flux_s = constitutive_flux(flux_s, forchheimer_inverse_flux_scale)
    area_s = mesh["radius_s_face"][1:-1] * dx
    tendency[:-1] -= flux_s * area_s / volume[:-1]
    tendency[1:] += flux_s * area_s / volume[1:]
    coefficient_x_face = 0.5 * (coefficient_x[:, 1:] + coefficient_x[:, :-1])
    flux_x = -coefficient_x_face * (field[:, 1:] - field[:, :-1]) / dx
    if forchheimer_inverse_flux_scale > 0.0:
        flux_x = constitutive_flux(flux_x, forchheimer_inverse_flux_scale)
    area_x = mesh["radius_x_face"][:, 1:-1] * ds[:, None]
    tendency[:, :-1] -= flux_x * area_x / volume[:, :-1]
    tendency[:, 1:] += flux_x * area_x / volume[:, 1:]
    return tendency


def _quasi_one_dimensional_flow(z, geometry, mission):
    area_ratio = (geometry.inner_radius(z) / geometry.throat_radius) ** 2
    mach = np.array([_supersonic_mach(value) for value in area_ratio])
    gamma = 1.22
    factor = 1.0 + 0.5 * (gamma - 1.0) * mach**2
    return {
        "mach": mach,
        "temperature": mission.stagnation_temperature / factor,
        "pressure": mission.chamber_pressure / factor ** (gamma / (gamma - 1.0)),
    }


def _supersonic_mach(area_ratio: float, gamma: float = 1.22) -> float:
    if area_ratio <= 1.0 + 1e-10:
        return 1.0
    low, high = 1.0, 10.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        value = (1.0 / middle) * (
            (2.0 / (gamma + 1.0))
            * (1.0 + 0.5 * (gamma - 1.0) * middle**2)
        ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        if value < area_ratio:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _hot_fire_duty(time_value: float, mission: NozzleMission) -> float:
    ramp = np.clip(time_value / mission.ramp_time, 0.0, 1.0)
    shutdown = np.clip((mission.duration - time_value) / mission.shutdown_time, 0.0, 1.0)
    return float((3.0 * ramp**2 - 2.0 * ramp**3) * (3.0 * shutdown**2 - 2.0 * shutdown**3))


def _nozzle_stress(temperature, carbon, crack, gas_pressure, mesh, geometry, mechanics):
    hot_fraction = np.clip((temperature - 293.15) / (1873.15 - 293.15), 0.0, 1.0)
    expansion = mechanics.thermal_expansion_cold + hot_fraction * (
        mechanics.thermal_expansion_hot - mechanics.thermal_expansion_cold
    )
    effective_modulus = mechanics.young_modulus * np.clip(carbon * (1.0 - crack), 0.08, 1.0)
    mean_temperature = np.mean(temperature, axis=1, keepdims=True)
    thermal_mismatch = np.abs(temperature - mean_temperature) + mechanics.restraint_fraction * np.maximum(
        mean_temperature - mechanics.reference_temperature, 0.0
    )
    thermal = effective_modulus * expansion * thermal_mismatch / (1.0 - mechanics.poisson_ratio)
    hoop_surface = gas_pressure * mesh["r_inner"] / geometry.wall_thickness
    depth = np.linspace(0.0, 1.0, temperature.shape[1])[None, :]
    hoop = hoop_surface[:, None] * (1.0 - 0.15 * depth)
    return np.sqrt(thermal**2 + hoop**2)


def _crack_opening(temperature, stress, carbon, crack, transport, mechanics):
    thermal_closure = transport.high_temperature_closure * np.clip(
        (temperature - transport.closure_temperature) / transport.closure_temperature_span,
        0.0, 1.0,
    )
    return np.clip(
        1.0
        + transport.stress_opening_gain * np.clip(stress / mechanics.crack_threshold_stress, 0.0, 3.0)
        + transport.damage_opening_gain * crack
        + transport.carbon_loss_opening_gain * (1.0 - carbon)
        - thermal_closure,
        0.2, 12.0,
    )


def _validate(
    geometry,
    grid,
    mission,
    material,
    thermal_boundary,
    transport,
    chemistry,
    mechanics,
):
    for group in (geometry, grid, mission, material, transport, chemistry, mechanics):
        if not all(np.isfinite(float(value)) for value in asdict(group).values()):
            raise ValueError("all nozzle configuration values must be finite")
    boundary_values = (
        thermal_boundary.radius_exponent,
        thermal_boundary.bartz_area_exponent,
        thermal_boundary.gamma,
        thermal_boundary.prandtl,
        thermal_boundary.wall_reference_temperature,
    )
    if not all(np.isfinite(float(value)) for value in boundary_values):
        raise ValueError("all thermal-boundary values must be finite")
    if thermal_boundary.model not in {"legacy_radius_power", "bartz_area_profile"}:
        raise ValueError(f"unknown thermal boundary model {thermal_boundary.model!r}")
    if (
        thermal_boundary.radius_exponent <= 0.0
        or thermal_boundary.bartz_area_exponent <= 0.0
        or thermal_boundary.gamma <= 1.0
        or not 0.0 < thermal_boundary.prandtl <= 2.0
        or thermal_boundary.wall_reference_temperature <= 0.0
    ):
        raise ValueError("thermal-boundary parameters lie outside valid ranges")
    if geometry.length <= 0.0 or geometry.exit_radius <= 0.0 or geometry.wall_thickness <= 0.0:
        raise ValueError("nozzle dimensions must be positive")
    probe = np.linspace(0.0, geometry.length, 257)
    if np.any(np.diff(geometry.inner_radius(probe)) <= 0.0):
        raise ValueError("bell radius must increase monotonically")
    if grid.meridional_cells < 3 or grid.thickness_cells < 3:
        raise ValueError("nozzle grid needs at least three cells per direction")
    if grid.time_step <= 0.0 or grid.output_interval <= 0.0 or mission.duration <= 0.0:
        raise ValueError("time settings must be positive")
    if mission.chamber_pressure < 0.0:
        raise ValueError("chamber pressure must be non-negative")
    if mission.stagnation_temperature <= 0.0 or material.density <= 0.0:
        raise ValueError("stagnation temperature and density must be positive")
    if transport.meridional_to_through_diffusivity_ratio <= 0.0:
        raise ValueError("meridional-to-through diffusivity ratio must be positive")
    if transport.forchheimer_inverse_flux_scale < 0.0:
        raise ValueError("Forchheimer inverse flux scale must be non-negative")
    mesh = _shell_mesh(geometry, grid)
    maximum_diffusivity = (
        transport.steam_diffusivity_closed
        * transport.open_to_closed_diffusivity_ratio
        * max(1.0, transport.meridional_to_through_diffusivity_ratio)
    )
    maximum_thermal = max(material.conductivity_axial, material.conductivity_through) / (
        material.density * material.heat_capacity_ambient
    )
    explicit_number = grid.time_step * (
        max(maximum_diffusivity, maximum_thermal) / mesh["dx"]**2
        + max(maximum_diffusivity, maximum_thermal) / np.min(mesh["ds"])**2
    )
    if explicit_number >= 0.5:
        raise ValueError("time_step violates the explicit diffusion stability bound")


def _initial_nozzle_field(value, shape, *, name, lower, upper):
    field = np.asarray(value, dtype=float)
    if field.ndim == 0:
        field = np.full(shape, float(field), dtype=float)
    if field.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    if (
        not np.all(np.isfinite(field))
        or np.any(field < lower)
        or np.any(field > upper)
    ):
        raise ValueError(f"{name} must be finite and lie in [{lower}, {upper}]")
    return np.array(field, copy=True)


def _initial_surface_field(value, size, *, name, lower, upper):
    field = np.asarray(value, dtype=float)
    if field.ndim == 0:
        field = np.full(size, float(field), dtype=float)
    if field.shape != (size,):
        raise ValueError(f"{name} must be scalar or have shape {(size,)}")
    if (
        not np.all(np.isfinite(field))
        or np.any(field < lower)
        or np.any(field > upper)
    ):
        raise ValueError(f"{name} must be finite and lie in [{lower}, {upper}]")
    return np.array(field, copy=True)
