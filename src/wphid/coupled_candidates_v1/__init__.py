"""Declared physical and reaction-support candidates; no search or fitting."""

from .physical import Expression, ModulationKink, gas_modulation, thermal_modulation, exponential_barrier
from .model import RateSpec, SOURCE_CANDIDATES, assemble_candidate

__all__ = ["Expression", "ModulationKink", "gas_modulation", "thermal_modulation",
           "exponential_barrier", "RateSpec", "SOURCE_CANDIDATES", "assemble_candidate"]
