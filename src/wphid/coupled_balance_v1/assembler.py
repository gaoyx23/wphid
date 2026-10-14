"""Five-field balances with eleven linear source/storage/transport coefficients.

This assembles a declared two-path model, not an open mechanism estimator. All
external pressures, heat inputs, radiation temperatures and sealed sides must
be explicit. No unknown chemistry, material or transport truth is imported.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np

from .operators import SIDES, Modulation, WeakProjector, attenuated_flux, finite, heat_primitive


FIELDS = ("carbon", "pO2", "pH2O", "pProducts", "T")
PRESSURES = FIELDS[1:4]
COLUMNS = ("carbon_rate_scale", "oxygen_response", "steam_response", "product_response",
           "heat_response", "gas_scale", "surface_transfer", "thermal_scale",
           "heat_input_scale", "radiation_scale", "heat_capacity_contrast")


@dataclass(frozen=True)
class RatePair:
    shapes: np.ndarray
    jacobian: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportShape:
    direction_ratio: float
    steam_ratio: float
    product_ratio: float
    forchheimer_product: float

    def __post_init__(self):
        positive = (self.direction_ratio, self.steam_ratio, self.product_ratio)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("transport ratios must be positive and finite")
        if not np.isfinite(self.forchheimer_product) or self.forchheimer_product < 0:
            raise ValueError("Forchheimer product must be nonnegative and finite")


@dataclass(frozen=True)
class BoundaryInputs:
    pressure_external: Mapping[tuple[str, str], np.ndarray]
    pressure_barriers: Mapping[tuple[str, str], Modulation]
    sealed_pressure: frozenset[tuple[str, str]]
    heat_inward: Mapping[str, np.ndarray]
    radiation_ambient: Mapping[str, np.ndarray]
    sealed_heat: frozenset[str]


@dataclass(frozen=True)
class WeakSystem:
    target: np.ndarray
    matrix: np.ndarray
    matrix_jacobian: Mapping[str, np.ndarray]
    row_keys: tuple[tuple[str, int, int], ...]
    column_keys: tuple[str, ...] = COLUMNS

    def residual(self, coefficients):
        coefficients = finite(coefficients, (len(self.column_keys),), "linear coefficients")
        return self.target-self.matrix @ coefficients

    def parameter_derivatives(self, coefficients):
        coefficients = finite(coefficients, (len(self.column_keys),), "linear coefficients")
        return {name: value @ coefficients for name, value in self.matrix_jacobian.items()}


def arrhenius_pair(states, *, oxygen_energy, steam_energy, oxygen_order, steam_order,
                   carbon_order, gas_constant=8.314462618):
    parameters = (oxygen_energy, steam_energy, oxygen_order, steam_order, carbon_order, gas_constant)
    if not all(np.isfinite(v) for v in parameters) or min(parameters) <= 0:
        raise ValueError("positive finite kinetic parameters and gas constant are required")
    temperature, oxygen, steam, carbon = (np.asarray(states[k], dtype=float)
                                          for k in ("T", "pO2", "pH2O", "carbon"))
    shape = temperature.shape
    for name, array in (("T", temperature), ("oxygen", oxygen), ("steam", steam), ("carbon", carbon)):
        finite(array, shape, name)
    if np.any(temperature <= 0) or np.any(carbon <= 0) or np.any(oxygen < 0) or np.any(steam < 0):
        raise ValueError("state outside the declared kinetic domain")
    fo = np.exp(-oxygen_energy/(gas_constant*temperature))*(oxygen/2e4)**oxygen_order*carbon**carbon_order
    fw = np.exp(-steam_energy/(gas_constant*temperature))*(steam/1e6)**steam_order*carbon**carbon_order
    log_o, log_w = np.zeros_like(oxygen), np.zeros_like(steam)
    np.log(oxygen/2e4, out=log_o, where=oxygen > 0)
    np.log(steam/1e6, out=log_w, where=steam > 0)
    zero = np.zeros_like(fo)
    jacobian = {
        "kinetic.oxygen_E_1e5": np.stack((-1e5*fo/(gas_constant*temperature), zero)),
        "kinetic.steam_E_1e5": np.stack((zero, -1e5*fw/(gas_constant*temperature))),
        "kinetic.oxygen_order": np.stack((fo*log_o, zero)),
        "kinetic.steam_order": np.stack((zero, fw*log_w)),
        "kinetic.carbon_order": np.stack((fo*np.log(carbon), fw*np.log(carbon))),
    }
    return RatePair(np.stack((fo, fw)), jacobian)


def assemble_balances(projector: WeakProjector, states, rates: RatePair, *, relative_rate,
                      transport: TransportShape, gas_modulation: Modulation,
                      thermal_modulation: Modulation, thermal_direction_ratio,
                      heat_knots, boundary: BoundaryInputs):
    if set(states) != set(FIELDS):
        raise ValueError("exactly the five declared balance fields are required")
    fields = {key: finite(states[key], projector.state_shape, key) for key in FIELDS}
    if not np.isfinite(relative_rate) or relative_rate <= 0:
        raise ValueError("two-path relative rate must be positive and finite")
    if not np.isfinite(thermal_direction_ratio) or thermal_direction_ratio <= 0:
        raise ValueError("thermal direction ratio must be positive and finite")
    shapes = finite(rates.shapes, (2, *projector.state_shape), "rate shapes")
    if np.any(shapes < 0):
        raise ValueError("physical rate shapes cannot be negative")
    if any(not isinstance(key, str) or not key for key in rates.jacobian):
        raise ValueError("rate derivative names must be nonempty strings")
    rate_jacobian = {key: finite(value, shapes.shape, "rate derivative")
                     for key, value in rates.jacobian.items()}
    gas_value, gas_jacobian = gas_modulation.checked(projector.state_shape, "gas modulation")
    thermal_value, thermal_jacobian = thermal_modulation.checked(projector.state_shape, "thermal modulation")
    active_pressure = set(boundary.pressure_external)
    declared_pressure = {(key, side) for key in PRESSURES for side in SIDES}
    if (active_pressure & set(boundary.sealed_pressure) or
        active_pressure | set(boundary.sealed_pressure) != declared_pressure or
        set(boundary.pressure_barriers) != active_pressure):
        raise ValueError("every pressure boundary must be explicitly active or sealed, with its barrier")
    heat_active = set(boundary.heat_inward) | set(boundary.radiation_ambient)
    if (heat_active & set(boundary.sealed_heat) or
        heat_active | set(boundary.sealed_heat) != set(SIDES)):
        raise ValueError("every energy boundary must be explicitly forced/radiating or sealed")
    block_rows = int(np.prod(projector.row_shape))
    matrix = np.zeros((len(FIELDS)*block_rows, len(COLUMNS)))
    derivatives = {}
    field_slices = {key: slice(i*block_rows, (i+1)*block_rows) for i, key in enumerate(FIELDS)}

    def put(key, column, values, derivative=None):
        target = matrix if derivative is None else derivatives.setdefault(derivative, np.zeros_like(matrix))
        target[field_slices[key], COLUMNS.index(column)] += values

    mixture = shapes[0]+relative_rate*shapes[1]

    def source_pair(pair, name=None, mixture_only=False):
        combined = pair[0]+relative_rate*pair[1] if not mixture_only else pair[0]
        for key, column, sign in (("carbon", "carbon_rate_scale", -1),
                                  ("pProducts", "product_response", 1), ("T", "heat_response", 1)):
            put(key, column, sign*projector.source(combined), name)
        if not mixture_only:
            put("pO2", "oxygen_response", -projector.source(pair[0]), name)
            put("pH2O", "steam_response", -projector.source(pair[1]), name)

    source_pair(shapes)
    for name, pair in rate_jacobian.items():
        source_pair(pair, name)
    source_pair(np.stack((shapes[1], np.zeros_like(mixture))), "reaction.relative_rate", mixture_only=True)
    gas_faces = projector.face_modulation(gas_value)
    gas_derivative_faces = {name: projector.face_modulation(value) for name, value in gas_jacobian.items()}
    ratios = {"pO2": 1., "pH2O": transport.steam_ratio, "pProducts": transport.product_ratio}
    for key in PRESSURES:
        gradient_s, gradient_x = projector.gradient(fields[key])
        ratio = ratios[key]
        base_s = -ratio*transport.direction_ratio*gas_faces[0]*gradient_s
        base_x = -ratio*gas_faces[1]*gradient_x
        flux_s, dv_s, db_s = attenuated_flux(base_s, transport.forchheimer_product)
        flux_x, dv_x, db_x = attenuated_flux(base_x, transport.forchheimer_product)
        put(key, "gas_scale", projector.flux(flux_s, flux_x))
        put(key, "gas_scale", projector.flux(db_s, db_x), "gas.forchheimer_product")
        put(key, "gas_scale", projector.flux(dv_s*base_s/transport.direction_ratio, np.zeros_like(flux_x)),
            "gas.direction_ratio")
        if key != "pO2":
            name = "gas.steam_ratio" if key == "pH2O" else "gas.product_ratio"
            put(key, "gas_scale", projector.flux(dv_s*base_s/ratio, dv_x*base_x/ratio), name)
        for name, faces in gas_derivative_faces.items():
            ds = -ratio*transport.direction_ratio*faces[0]*gradient_s
            dx = -ratio*faces[1]*gradient_x
            put(key, "gas_scale", projector.flux(dv_s*ds, dv_x*dx), name)
    for (key, side), external in boundary.pressure_external.items():
        trace = projector.trace(fields[key], side)
        external = finite(external, trace.shape, "external pressure")
        barrier, barrier_jacobian = boundary.pressure_barriers[(key, side)].checked(trace.shape, "surface barrier")
        drive = external-trace
        put(key, "surface_transfer", projector.boundary(drive*barrier, side))
        for name, value in barrier_jacobian.items():
            put(key, "surface_transfer", projector.boundary(drive*value, side), name)
    gradient_s, gradient_x = projector.gradient(fields["T"])
    thermal_faces = projector.face_modulation(thermal_value)
    flux_s = -thermal_direction_ratio*thermal_faces[0]*gradient_s
    flux_x = -thermal_faces[1]*gradient_x
    put("T", "thermal_scale", projector.flux(flux_s, flux_x))
    put("T", "thermal_scale", projector.flux(flux_s/thermal_direction_ratio, np.zeros_like(flux_x)),
        "thermal.direction_ratio")
    for name, value in thermal_jacobian.items():
        face_s, face_x = projector.face_modulation(value)
        put("T", "thermal_scale", projector.flux(-thermal_direction_ratio*face_s*gradient_s, -face_x*gradient_x), name)
    for side, value in boundary.heat_inward.items():
        shape = projector.trace(fields["T"], side).shape
        put("T", "heat_input_scale", projector.boundary(finite(value, shape, "inward heat input"), side))
    for side, value in boundary.radiation_ambient.items():
        trace = projector.trace(fields["T"], side)
        ambient = finite(value, trace.shape, "ambient temperature")
        if np.any(ambient < 0) or np.any(trace < 0):
            raise ValueError("radiation temperatures cannot be negative")
        put("T", "radiation_scale", projector.boundary(-(trace**4-ambient**4), side))
    primitive, _ = heat_primitive(fields["T"], heat_knots)
    put("T", "heat_capacity_contrast", -projector.storage(primitive))
    target = np.concatenate([projector.storage(fields[key]) for key in FIELDS])
    row_keys = tuple((key, window, spatial) for key in FIELDS
                     for window in range(projector.row_shape[0])
                     for spatial in range(projector.row_shape[1]))
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("nonfinite weak assembly")
    return WeakSystem(target, matrix, derivatives, row_keys)
