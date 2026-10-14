"""A separable source representation, without fixed truth conversions or fits.

Two nonnegative rate shapes share a positive relative rate scale. Carbon,
product, and thermal sources share their mixture; oxygen and steam conversion
amplitudes remain free. Source shape assumptions still need model qualification.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


FIELD_KEYS = ("carbon", "pO2", "pH2O", "pProducts", "T")
COEFFICIENT_KEYS = (
    "carbon_rate_scale", "oxygen_response", "steam_response",
    "product_response", "heat_response",
)


@dataclass(frozen=True)
class BundleDesign:
    matrix: np.ndarray
    derivative_relative_rate: np.ndarray
    derivative_heat_contrast: np.ndarray
    coefficient_keys: tuple[str, ...]
    row_keys: tuple[tuple[str, int], ...]
    carbon_scale_anchor_present_in_model: bool


def build_bundle_design(weak_weights: Mapping, rate_shapes, *, relative_rate,
                        heat_fraction, heat_contrast):
    """Construct source columns and their two outer-coordinate derivatives.

    Each weak_weights[field] has shape (windows, nodes). rate_shapes has shape
    (2, nodes), excluding both rate amplitudes. heat_fraction is the declared
    heat-capacity interpolation function, not truth heat-capacity values.
    Unknown conversions are the linear coefficients, not inputs to this function.
    No noise correction, PDE transport assembly, optimization or selection occurs.
    """
    rates = np.asarray(rate_shapes, dtype=float)
    fraction = np.asarray(heat_fraction, dtype=float)
    if rates.ndim != 2 or rates.shape[0] != 2 or rates.shape[1] == 0:
        raise ValueError("two nonempty rate shapes are required")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("physical rate shapes must be finite and nonnegative")
    if not np.isfinite(relative_rate) or relative_rate <= 0:
        raise ValueError("relative_rate must be positive and finite")
    if not np.isfinite(heat_contrast) or heat_contrast <= -1:
        raise ValueError("heat contrast must keep heat capacity positive")
    if fraction.shape != (rates.shape[1],) or not np.all(np.isfinite(fraction)):
        raise ValueError("heat_fraction has the wrong shape or nonfinite entries")
    if np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError("heat_fraction must lie in the declared [0, 1] interval")
    if not isinstance(weak_weights, Mapping) or not weak_weights:
        raise ValueError("at least one named source field is required")
    if any(field not in FIELD_KEYS for field in weak_weights):
        raise ValueError("unqualified source field; crack/stress cannot be tied here")
    mixture = rates[0] + relative_rate * rates[1]
    capacity_shape = 1 + heat_contrast * fraction
    sources = (-mixture, -rates[0], -rates[1], mixture, mixture / capacity_shape)
    derivatives_ratio = (-rates[1], np.zeros_like(mixture), np.zeros_like(mixture),
                         rates[1], rates[1] / capacity_shape)
    derivatives_heat = (np.zeros_like(mixture),) * 4 + (
        -mixture * fraction / capacity_shape**2,)
    fields = tuple(field for field in FIELD_KEYS if field in weak_weights)
    coefficients = tuple(COEFFICIENT_KEYS[FIELD_KEYS.index(field)] for field in fields)
    weights = {}
    row_keys = []
    for field in fields:
        matrix = np.asarray(weak_weights[field], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != rates.shape[1]:
            raise ValueError("weak source weights must be nonempty windows by nodes")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("weak source weights must be finite")
        weights[field] = matrix
        row_keys.extend((field, window) for window in range(matrix.shape[0]))
    result = np.zeros((len(row_keys), len(fields)))
    ratio_derivative = np.zeros_like(result)
    heat_derivative = np.zeros_like(result)
    start = 0
    for column, field in enumerate(fields):
        index = FIELD_KEYS.index(field)
        weight = weights[field]
        stop = start + weight.shape[0]
        result[start:stop, column] = weight @ sources[index]
        ratio_derivative[start:stop, column] = weight @ derivatives_ratio[index]
        heat_derivative[start:stop, column] = weight @ derivatives_heat[index]
        start = stop
    return BundleDesign(
        matrix=result, derivative_relative_rate=ratio_derivative,
        derivative_heat_contrast=heat_derivative,
        coefficient_keys=coefficients, row_keys=tuple(row_keys),
        carbon_scale_anchor_present_in_model="carbon" in fields,
    )


def decode_source_coefficients(design: BundleDesign, coefficients, *, relative_rate):
    """Decode positive fitted source coefficients conditional on a carbon anchor.

    This is an algebraic coordinate conversion, not evidence of actual parameter
    recovery. Density, absolute heat capacity and heat release remain inseparable.
    An unknown carbon measurement gain invalidates the absolute scale interpretation.
    """
    values = np.asarray(coefficients, dtype=float)
    if values.shape != (len(design.coefficient_keys),) or not np.all(np.isfinite(values)):
        raise ValueError("source coefficients have the wrong shape or nonfinite entries")
    if np.any(values <= 0) or not np.isfinite(relative_rate) or relative_rate <= 0:
        raise ValueError("active two-path interpretation requires positive scales")
    if not design.carbon_scale_anchor_present_in_model:
        raise ValueError("absolute rate scales require a qualified carbon-unit anchor")
    mapping = dict(zip(design.coefficient_keys, values))
    amplitude = mapping["carbon_rate_scale"]
    decoded = {"oxygen_rate_scale": amplitude, "steam_rate_scale": amplitude * relative_rate}
    if "oxygen_response" in mapping:
        decoded["oxygen_conversion"] = mapping["oxygen_response"] / amplitude
    if "steam_response" in mapping:
        decoded["steam_conversion"] = mapping["steam_response"] / (amplitude * relative_rate)
    if "product_response" in mapping:
        decoded["product_conversion"] = mapping["product_response"] / amplitude
    if "heat_response" in mapping:
        decoded["heat_release_over_density_ambient_capacity"] = mapping["heat_response"] / amplitude
    return decoded
