"""Reduced SiC/SiO2 surface-layer kinetics for nozzle truth extensions.

The model represents literature-supported competition between passive silica
formation, steam-driven silica removal, stress-assisted spallation, and active
SiC recession.  Default coefficients are synthetic calibration values; they
are not claimed as HM7 material measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

import numpy as np


GAS_CONSTANT = 8.314462618


@dataclass(frozen=True)
class NozzleSurfaceChemistry:
    """Equivalent inner-wall SiO2-layer and recession parameters."""

    initial_silica_thickness: float = 40.0e-6
    maximum_silica_thickness: float = 120.0e-6
    silica_barrier_strength: float = 3.0
    passive_formation_pre_exponential: float = 1.0e-2
    passive_formation_activation_energy: float = 180_000.0
    passive_formation_oxygen_exponent: float = 0.5
    reference_oxygen_pressure: float = 1_000.0
    steam_volatilization_pre_exponential: float = 3.0e-3
    steam_volatilization_activation_energy: float = 160_000.0
    steam_volatilization_exponent: float = 0.75
    reference_steam_pressure: float = 1.0e6
    active_transition_temperature: float = 1_700.0
    active_transition_span: float = 120.0
    active_transition_oxygen_pressure: float = 1_500.0
    active_transition_oxygen_exponent: float = 2.0
    spallation_rate_scale: float = 1.0e-8
    spallation_threshold_stress: float = 75.0e6
    spallation_stress_exponent: float = 2.0
    active_recession_pre_exponential: float = 2.0e-4
    active_recession_activation_energy: float = 160_000.0
    carbon_recession_per_fraction: float = 2.0e-4
    fixed_domain_recession_fraction_limit: float = 0.05


@lru_cache(maxsize=32)
def validate_surface_chemistry(parameters: NozzleSurfaceChemistry) -> None:
    values = asdict(parameters)
    if not all(np.isfinite(float(value)) for value in values.values()):
        raise ValueError("surface-chemistry parameters must be finite")
    strictly_positive = (
        parameters.maximum_silica_thickness,
        parameters.passive_formation_activation_energy,
        parameters.reference_oxygen_pressure,
        parameters.steam_volatilization_activation_energy,
        parameters.reference_steam_pressure,
        parameters.active_transition_temperature,
        parameters.active_transition_span,
        parameters.active_transition_oxygen_pressure,
        parameters.spallation_threshold_stress,
        parameters.active_recession_activation_energy,
    )
    if min(strictly_positive) <= 0.0:
        raise ValueError("surface-chemistry scales must be positive")
    nonnegative = (
        parameters.initial_silica_thickness,
        parameters.silica_barrier_strength,
        parameters.passive_formation_pre_exponential,
        parameters.passive_formation_oxygen_exponent,
        parameters.steam_volatilization_pre_exponential,
        parameters.steam_volatilization_exponent,
        parameters.active_transition_oxygen_exponent,
        parameters.spallation_rate_scale,
        parameters.spallation_stress_exponent,
        parameters.active_recession_pre_exponential,
        parameters.carbon_recession_per_fraction,
    )
    if min(nonnegative) < 0.0:
        raise ValueError("surface-chemistry rates and exponents must be non-negative")
    if parameters.initial_silica_thickness > parameters.maximum_silica_thickness:
        raise ValueError("initial silica thickness cannot exceed its maximum")
    if not 0.0 < parameters.fixed_domain_recession_fraction_limit < 1.0:
        raise ValueError("fixed-domain recession limit must lie in (0, 1)")


def csic_surface_rates(
    temperature: np.ndarray,
    oxygen_pressure: np.ndarray,
    steam_pressure: np.ndarray,
    stress: np.ndarray,
    silica_thickness: np.ndarray,
    underlying_carbon_reaction_rate: np.ndarray,
    parameters: NozzleSurfaceChemistry | None = None,
) -> Mapping[str, np.ndarray]:
    """Evaluate the reduced surface-layer rates at meridional locations.

    Rate units for formation, volatilization, spallation, and recession are
    metres per second.  ``active_oxidation_fraction`` and ``barrier_factor``
    are dimensionless algebraic states.
    """

    parameters = NozzleSurfaceChemistry() if parameters is None else parameters
    validate_surface_chemistry(parameters)
    arrays = np.broadcast_arrays(
        np.asarray(temperature, dtype=float),
        np.asarray(oxygen_pressure, dtype=float),
        np.asarray(steam_pressure, dtype=float),
        np.asarray(stress, dtype=float),
        np.asarray(silica_thickness, dtype=float),
        np.asarray(underlying_carbon_reaction_rate, dtype=float),
    )
    temperature, oxygen, steam, stress, silica, carbon_rate = arrays
    safe_temperature = np.maximum(temperature, 1.0)
    coverage = np.clip(
        silica / parameters.maximum_silica_thickness,
        0.0,
        1.0,
    )
    oxygen_activity = (
        np.maximum(oxygen, 0.0) / parameters.reference_oxygen_pressure
    ) ** parameters.passive_formation_oxygen_exponent
    steam_activity = (
        np.maximum(steam, 0.0) / parameters.reference_steam_pressure
    ) ** parameters.steam_volatilization_exponent
    thermal_active = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (safe_temperature - parameters.active_transition_temperature)
                / parameters.active_transition_span,
                -60.0,
                60.0,
            )
        )
    )
    oxygen_suppression = 1.0 / (
        1.0
        + (
            np.maximum(oxygen, 0.0)
            / parameters.active_transition_oxygen_pressure
        ) ** parameters.active_transition_oxygen_exponent
    )
    active_fraction = np.clip(thermal_active * oxygen_suppression, 0.0, 1.0)

    formation = (
        parameters.passive_formation_pre_exponential
        * np.exp(
            -parameters.passive_formation_activation_energy
            / (GAS_CONSTANT * safe_temperature)
        )
        * oxygen_activity
        * (1.0 - coverage)
        * (1.0 - active_fraction)
    )
    volatilization = (
        parameters.steam_volatilization_pre_exponential
        * np.exp(
            -parameters.steam_volatilization_activation_energy
            / (GAS_CONSTANT * safe_temperature)
        )
        * steam_activity
        * coverage
    )
    stress_excess = np.maximum(
        stress / parameters.spallation_threshold_stress - 1.0,
        0.0,
    )
    spallation = (
        parameters.spallation_rate_scale
        * stress_excess**parameters.spallation_stress_exponent
        * coverage
    )
    active_recession = (
        parameters.active_recession_pre_exponential
        * np.exp(
            -parameters.active_recession_activation_energy
            / (GAS_CONSTANT * safe_temperature)
        )
        * active_fraction
        * (1.0 - coverage)
        * (oxygen_activity + steam_activity)
    )
    carbon_recession = (
        parameters.carbon_recession_per_fraction
        * np.maximum(carbon_rate, 0.0)
    )
    barrier = np.exp(-parameters.silica_barrier_strength * coverage)
    result = {
        "passive_formation_rate": np.asarray(formation),
        "steam_volatilization_rate": np.asarray(volatilization),
        "stress_spallation_rate": np.asarray(spallation),
        "silica_net_rate": np.asarray(formation - volatilization - spallation),
        "active_recession_rate": np.asarray(active_recession),
        "carbon_recession_rate": np.asarray(carbon_recession),
        "total_recession_rate": np.asarray(active_recession + carbon_recession),
        "active_oxidation_fraction": np.asarray(active_fraction),
        "barrier_factor": np.asarray(barrier),
    }
    for values in result.values():
        values.setflags(write=False)
    return MappingProxyType(result)
