"""Shared physical syntax for transport proposals and candidate forwarding.

Candidate objects encode only declared hypotheses, never truth labels. The
state-dependent gain, flux nonlinearity, and directional ratio are separate
axes. This is a restricted transport library, not full five-mechanism recovery.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

MODULATIONS = ("combined_opening", "constant", "crack_only", "carbon_loss_only", "stress_only")


def constitutive_factor(linear_flux, inverse_flux_scale):
    """Positive multiplier solving the signed Darcy--Forchheimer relation."""
    alpha = float(inverse_flux_scale)
    flux = np.asarray(linear_flux, dtype=float)
    if not math.isfinite(alpha) or alpha < 0 or not np.all(np.isfinite(flux)):
        raise ValueError("finite flux and nonnegative inverse flux scale required")
    if alpha == 0:
        return np.ones_like(flux)
    return 2. / (1. + np.sqrt(1. + 4. * alpha * np.abs(flux)))


def constitutive_flux(linear_flux, inverse_flux_scale):
    """Solve J + alpha*abs(J)*J = J_linear, preserving signed flux."""
    flux = np.asarray(linear_flux, dtype=float)
    return flux * constitutive_factor(flux, inverse_flux_scale)


@dataclass(frozen=True)
class TransportCandidate:
    modulation: str
    flux_law: str
    directional_ratio: float = 1.0
    inverse_flux_scale: float = 0.0

    def __post_init__(self):
        ratio, alpha = float(self.directional_ratio), float(self.inverse_flux_scale)
        if self.modulation not in MODULATIONS or self.flux_law not in ("Darcy", "Forchheimer"):
            raise ValueError("unsupported transport hypothesis")
        if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("direction ratio must be positive and inverse flux scale nonnegative")
        if (self.flux_law == "Darcy" and alpha != 0) or (self.flux_law == "Forchheimer" and alpha <= 0):
            raise ValueError("flux law and inverse flux scale are inconsistent")
        object.__setattr__(self, "directional_ratio", ratio)
        object.__setattr__(self, "inverse_flux_scale", alpha)

    @property
    def candidate_id(self):
        return (f"transport-v1::{self.modulation}::{self.flux_law}"
                f"::ratio={self.directional_ratio.hex()}::alpha={self.inverse_flux_scale.hex()}")

    def configure_transport(self, base):
        """Preserve other supplied transport parameters; replace declared axes."""
        return replace(base, meridional_to_through_diffusivity_ratio=self.directional_ratio,
                       forchheimer_inverse_flux_scale=self.inverse_flux_scale)


def modulation_fraction(name, *, temperature, stress, carbon, crack, transport, mechanics):
    """Dimensionless modulation used identically in candidates and forward PDE.

Gain is 1 + (open_to_closed_ratio-1)*fraction. Alternatives share this bounded
amplitude convention; they are not automatically equivalent to old regressions
that estimated free intercepts and slopes.
"""
    if name not in MODULATIONS:
        raise ValueError("unsupported modulation")
    t, sig, phi, damage = np.broadcast_arrays(temperature, stress, carbon, crack)
    if not all(np.all(np.isfinite(a)) for a in (t, sig, phi, damage)):
        raise ValueError("state inputs must be finite")
    if name == "constant":
        return np.zeros_like(t, dtype=float)
    if name == "crack_only":
        return np.clip(damage, 0., 1.)
    if name == "carbon_loss_only":
        return np.clip(1. - phi, 0., 1.)
    if name == "stress_only":
        return np.clip(sig / mechanics.crack_threshold_stress / 3., 0., 1.)
    thermal_closure = transport.high_temperature_closure * np.clip(
        (t - transport.closure_temperature) / transport.closure_temperature_span, 0., 1.)
    opening = np.clip(1. + transport.stress_opening_gain * np.clip(
        sig / mechanics.crack_threshold_stress, 0., 3.)
        + transport.damage_opening_gain * damage
        + transport.carbon_loss_opening_gain * (1. - phi) - thermal_closure, .2, 12.)
    return np.clip((opening - 1.) / 9.6, 0., 1.)
