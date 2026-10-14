"""Support/response graph candidates with identical five-field row identities.

Inactive paths have no kinetic coordinates or source columns. Free source
responses use separate amplitudes; they are not silently forced into a common
mixture. Nuisance balances reuse the frozen conservative value assembler.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, vstack

from wphid.coupled_balance_v1.assembler import (
    BoundaryInputs, COLUMNS, FIELDS, PRESSURES, RatePair, WeakSystem, assemble_balances,
)
from wphid.coupled_balance_v1.noise import SharedObservationLinearization
from wphid.coupled_balance_v1.operators import Modulation, attenuated_flux, heat_primitive
from wphid.coupled_observation_v1.linearization import (
    SourceDerivativeFailure, SparseWeakMaps, clean, scale_rows,
)
from .physical import Expression


SOURCE_CANDIDATES = {
    "null": (), "oxygen_only": ("oxygen",), "steam_only": ("steam",),
    "both_shared": ("oxygen", "steam"), "both_free": ("oxygen", "steam"),
    "both_swapped_product": ("oxygen", "steam"), "both_oxygen_heat": ("oxygen", "steam"),
}
NUISANCE_COLUMNS = COLUMNS[5:]


@dataclass(frozen=True)
class RateSpec:
    energy: float
    pressure_order: float
    carbon_order: float

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in (self.energy, self.pressure_order, self.carbon_order)):
            raise ValueError("active rates require positive finite parameters")


def _rate(path, spec, states, maps, shared_carbon):
    pressure_key, pressure_scale = ("pO2", 2e4) if path == "oxygen" else ("pH2O", 1e6)
    temperature, carbon, pressure = (states[key].value.ravel() for key in ("T", "carbon", pressure_key))
    if np.any(temperature <= 0) or np.any(carbon <= 0) or np.any(pressure < 0):
        raise ValueError("active rate outside declared kinetic domain")
    R = 8.314462618
    value = np.exp(-spec.energy/(R*temperature))*(pressure/pressure_scale)**spec.pressure_order*carbon**spec.carbon_order
    zero = pressure == 0
    variable = np.asarray(states[pressure_key].jacobian.getnnz(axis=1)).ravel() > 0
    active_zero = zero & variable & maps.source_used
    if spec.pressure_order < 1 and np.any(active_zero):
        raise SourceDerivativeFailure({"field": pressure_key, "active_path": path,
            "unbounded_active_derivative_nodes": int(np.count_nonzero(active_zero))})
    slope = np.zeros_like(value)
    np.divide(spec.pressure_order*value, pressure, out=slope, where=pressure > 0)
    if spec.pressure_order == 1:
        slope[zero] = np.exp(-spec.energy/(R*temperature[zero]))*carbon[zero]**spec.carbon_order/pressure_scale
    derivative = scale_rows(states["T"].jacobian, value*spec.energy/(R*temperature**2))
    derivative += scale_rows(states["carbon"].jacobian, value*spec.carbon_order/carbon)
    derivative += scale_rows(states[pressure_key].jacobian, slope)
    log_pressure = np.zeros_like(pressure)
    np.log(pressure/pressure_scale, out=log_pressure, where=pressure > 0)
    qname = "kinetic.carbon_order" if shared_carbon else "kinetic."+path+".carbon_order"
    shape = states["T"].value.shape
    parameters = {
        "kinetic."+path+".energy_1e5": (-1e5*value/(R*temperature)).reshape(shape),
        "kinetic."+path+".pressure_order": (value*log_pressure).reshape(shape),
        qname: (value*np.log(carbon)).reshape(shape),
    }
    report = {"zero_pressure_nodes": int(np.count_nonzero(zero)),
              "zero_pressure_nodes_outside_source": int(np.count_nonzero(zero & ~maps.source_used)),
              "zero_pressure_nodes_in_storage": int(np.count_nonzero(zero &
                  (np.asarray(maps.storage.getnnz(axis=0)).ravel() > 0))),
              "observation_law_qualified": False}
    return Expression(value.reshape(shape), clean(derivative), parameters), report


def _sources(candidate, rates, reference, relative_rate):
    sources = []
    if candidate == "null":
        return sources
    if candidate == "both_free":
        for path, pressure_field in (("oxygen", "pO2"), ("steam", "pH2O")):
            rate = rates[path]
            sources.extend((("carbon", "carbon."+path+"_rate_scale", -rate),
                (pressure_field, path+"_response", -rate),
                ("pProducts", "product."+path+"_response", rate),
                ("T", "heat."+path+"_response", rate)))
        return sources
    if len(rates) == 1:
        mixture = next(iter(rates.values()))
    else:
        ratio = Expression.constant(relative_rate, reference, parameter="reaction.relative_rate")
        mixture = rates["oxygen"]+ratio*rates["steam"]
    sources.append(("carbon", "carbon_rate_scale", -mixture))
    for path, field in (("oxygen", "pO2"), ("steam", "pH2O")):
        if path in rates:
            sources.append((field, path+"_response", -rates[path]))
    product, heat = mixture, mixture
    if candidate == "both_swapped_product":
        product = ratio*rates["oxygen"]+rates["steam"]
    if candidate == "both_oxygen_heat":
        heat = rates["oxygen"]
    sources.extend((("pProducts", "product_response", product), ("T", "heat_response", heat)))
    return sources


def assemble_candidate(projector, layout, states, *, source_candidate, rate_specs, relative_rate,
                       shared_carbon_order, transport, flux_law, gas, thermal, thermal_direction_ratio,
                       heat_knots, boundary, barrier_parameters=None, maps=None):
    if source_candidate not in SOURCE_CANDIDATES:
        raise ValueError("unsupported source graph")
    if (flux_law not in ("Darcy", "Forchheimer") or
            flux_law == "Darcy" and transport.forchheimer_product != 0 or
            flux_law == "Forchheimer" and transport.forchheimer_product <= 0):
        raise ValueError("declared flux law and product coordinate disagree")
    active = SOURCE_CANDIDATES[source_candidate]
    if set(rate_specs) != set(active):
        raise ValueError("only active path parameters may be provided")
    uses_ratio = len(active) == 2 and source_candidate != "both_free"
    if uses_ratio:
        if relative_rate is None or not np.isfinite(relative_rate) or relative_rate <= 0:
            raise ValueError("positive shared mixture ratio required")
    elif relative_rate is not None:
        raise ValueError("null, single and free graphs have no shared ratio coordinate")
    if shared_carbon_order and len(active) == 2 and rate_specs[active[0]].carbon_order != rate_specs[active[1]].carbon_order:
        raise ValueError("declared shared carbon orders must agree")
    if set(states) != set(FIELDS) or any(item.jacobian.shape[1] != layout.size for item in states.values()):
        raise ValueError("five observed fields in the declared raw layout required")
    if gas.jacobian.shape[1] != layout.size or thermal.jacobian.shape[1] != layout.size:
        raise ValueError("constitutive maps must share raw coordinates")
    values = {key: item.value for key, item in states.items()}
    barrier_parameters = {} if barrier_parameters is None else barrier_parameters
    if not set(barrier_parameters).issubset(boundary.pressure_barriers):
        raise ValueError("barrier parameter derivatives need an active boundary")
    boundary_values = BoundaryInputs(
        {key: item.value for key, item in boundary.pressure_external.items()},
        {key: Modulation(item.value, barrier_parameters.get(key, {})) for key, item in boundary.pressure_barriers.items()},
        boundary.sealed_pressure, {key: item.value for key, item in boundary.heat_inward.items()},
        {key: item.value for key, item in boundary.radiation_ambient.items()}, boundary.sealed_heat)
    base = assemble_balances(projector, values, RatePair(np.zeros((2, *projector.state_shape))),
        relative_rate=1., transport=transport, gas_modulation=Modulation(gas.value, gas.parameters),
        thermal_modulation=Modulation(thermal.value, thermal.parameters),
        thermal_direction_ratio=thermal_direction_ratio, heat_knots=heat_knots, boundary=boundary_values)
    maps = SparseWeakMaps(projector) if maps is None else maps
    if maps.projector is not projector:
        raise ValueError("sparse maps belong to another projector")
    rate_expressions, domains = {}, {}
    for path in active:
        rate_expressions[path], domains[path] = _rate(path, rate_specs[path], states, maps, shared_carbon_order)
    sources = _sources(source_candidate, rate_expressions, Expression.observed(states["T"]), relative_rate)
    source_keys = tuple(column for _, column, _ in sources)
    columns = source_keys+tuple(NUISANCE_COLUMNS)
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate candidate coefficient identity")
    nrow, nraw = int(np.prod(projector.row_shape)), layout.size
    matrix = np.zeros((len(FIELDS)*nrow, len(columns)))
    indices = [COLUMNS.index(key) for key in NUISANCE_COLUMNS]
    matrix[:, len(source_keys):] = base.matrix[:, indices]
    parameters = {key: np.column_stack((np.zeros((len(base.target), len(source_keys))), value[:, indices]))
                  for key, value in base.matrix_jacobian.items() if key != "reaction.relative_rate" and
                  not (flux_law == "Darcy" and key == "gas.forchheimer_product")}
    blocks = [[csr_matrix((nrow, nraw)) for _ in FIELDS] for _ in columns]

    def put(field, column, derivative):
        blocks[columns.index(column)][FIELDS.index(field)] += derivative

    for field, column, expression in sources:
        row = slice(FIELDS.index(field)*nrow, (FIELDS.index(field)+1)*nrow)
        matrix[row, columns.index(column)] = maps.source @ expression.value.ravel()
        put(field, column, maps.source @ expression.jacobian)
        for key in sorted(expression.parameters):
            value = parameters.setdefault(key, np.zeros_like(matrix))
            value[row, columns.index(column)] = maps.source @ expression.parameters[key].ravel()
    ratios = {"pO2": 1., "pH2O": transport.steam_ratio, "pProducts": transport.product_ratio}
    for field in PRESSURES:
        for axis in ("s", "x"):
            direction = transport.direction_ratio if axis == "s" else 1.
            gradient = maps.gradient[axis] @ values[field].ravel()
            modulation = maps.average[axis] @ gas.value.ravel()
            _, slope, _ = attenuated_flux(-ratios[field]*direction*modulation*gradient, transport.forchheimer_product)
            derivative = scale_rows(maps.gradient[axis] @ states[field].jacobian, -ratios[field]*direction*modulation)
            derivative += scale_rows(maps.average[axis] @ gas.jacobian, -ratios[field]*direction*gradient)
            put(field, "gas_scale", maps.flux[axis] @ scale_rows(derivative, slope))
    for (field, side), external in boundary.pressure_external.items():
        if external.jacobian.shape[1] != nraw or boundary.pressure_barriers[(field, side)].jacobian.shape[1] != nraw:
            raise ValueError("boundary observations must share raw layout")
        trace = maps.trace[side] @ values[field].ravel()
        barrier = boundary.pressure_barriers[(field, side)]
        derivative = scale_rows(external.jacobian-maps.trace[side] @ states[field].jacobian, barrier.value)
        derivative += scale_rows(barrier.jacobian, external.value.ravel()-trace)
        put(field, "surface_transfer", maps.boundary[side] @ derivative)
    for axis in ("s", "x"):
        direction = thermal_direction_ratio if axis == "s" else 1.
        gradient = maps.gradient[axis] @ values["T"].ravel()
        modulation = maps.average[axis] @ thermal.value.ravel()
        derivative = scale_rows(maps.gradient[axis] @ states["T"].jacobian, -direction*modulation)
        derivative += scale_rows(maps.average[axis] @ thermal.jacobian, -direction*gradient)
        put("T", "thermal_scale", maps.flux[axis] @ derivative)
    for side, item in boundary.heat_inward.items():
        if item.jacobian.shape[1] != nraw:
            raise ValueError("heat observations must share raw layout")
        put("T", "heat_input_scale", maps.boundary[side] @ item.jacobian)
    for side, ambient in boundary.radiation_ambient.items():
        if ambient.jacobian.shape[1] != nraw:
            raise ValueError("radiation observations must share raw layout")
        trace = maps.trace[side] @ values["T"].ravel()
        derivative = scale_rows(maps.trace[side] @ states["T"].jacobian, -4*trace**3)
        derivative += scale_rows(ambient.jacobian, 4*ambient.value.ravel()**3)
        put("T", "radiation_scale", maps.boundary[side] @ derivative)
    _, fraction = heat_primitive(values["T"], heat_knots)
    put("T", "heat_capacity_contrast", -maps.storage @ scale_rows(states["T"].jacobian, fraction))
    target_jacobian = clean(vstack([maps.storage @ states[key].jacobian for key in FIELDS], format="csr"))
    linearization = SharedObservationLinearization(target_jacobian,
        tuple(clean(vstack(parts, format="csr")) for parts in blocks), layout.coordinate_keys)
    model = WeakSystem(base.target, matrix, {key: parameters[key] for key in sorted(parameters)}, base.row_keys, columns)
    return model, linearization, {"source_candidate": source_candidate, "active_paths": list(active),
        "flux_law": flux_law,
        "inactive_kinetic_coordinates_or_source_columns_retained": False, "source_domains": domains,
        "observation_law_or_calibration_qualified": False, "complete_open_mechanism_search": False}
