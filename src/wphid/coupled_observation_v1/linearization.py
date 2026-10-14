"""Analytic sparse raw-coordinate derivatives without fitting or noise draws.

Every data-dependent field, load and modulation supplies its observation map.
An explicit fixed value has a zero map. Zero-pressure source derivatives are
qualified against the actual source contraction, not by clipping observations.
Finite weak derivatives at a domain boundary do not qualify an observation law.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron, vstack

from wphid.coupled_balance_v1.assembler import (
    BoundaryInputs, COLUMNS, FIELDS, PRESSURES, arrhenius_pair, assemble_balances,
)
from wphid.coupled_balance_v1.noise import SharedObservationLinearization
from wphid.coupled_balance_v1.operators import (
    Modulation, SIDES, attenuated_flux, finite, heat_primitive,
)


def clean(value):
    matrix = csr_matrix(value)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("nonfinite observation derivative")
    return matrix


def scale_rows(matrix, weights):
    return clean(matrix.multiply(np.asarray(weights).ravel()[:, None]))


@dataclass(frozen=True)
class ObservedValue:
    value: np.ndarray
    jacobian: object

    def __post_init__(self):
        value = np.asarray(self.value, dtype=float)
        if not np.all(np.isfinite(value)) or value.size == 0:
            raise ValueError("observed value must be nonempty and finite")
        jacobian = clean(self.jacobian)
        if jacobian.shape[0] != value.size or jacobian.shape[1] < 1:
            raise ValueError("observation derivative has the wrong shape")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jacobian", jacobian)

    @classmethod
    def fixed(cls, value, coordinate_count):
        value = np.asarray(value, dtype=float)
        return cls(value, csr_matrix((value.size, coordinate_count)))


class ObservationLayout:
    """Unique raw blocks; repeated uses refer to the same selector columns."""

    def __init__(self, blocks):
        self.blocks, self.slices, self.size = {}, {}, 0
        for name, shape in blocks:
            shape = tuple(shape)
            if (not isinstance(name, str) or not name or name in self.blocks or not shape
                    or any(not isinstance(v, (int, np.integer)) or v < 1 for v in shape)):
                raise ValueError("raw blocks require unique names and positive integer shapes")
            count = int(np.prod(shape))
            self.blocks[name] = shape
            self.slices[name] = slice(self.size, self.size+count)
            self.size += count
        if not self.blocks:
            raise ValueError("at least one raw observation block is required")
        self.coordinate_keys = tuple(f"{name}[{i}]" for name, shape in self.blocks.items()
                                     for i in range(int(np.prod(shape))))

    def observed(self, name, value):
        shape = self.blocks[name]
        value = finite(value, shape, name)
        selection = self.slices[name]
        count = value.size
        jacobian = coo_matrix((np.ones(count), (np.arange(count),
                             np.arange(selection.start, selection.stop))),
                             shape=(count, self.size)).tocsr()
        return ObservedValue(value, jacobian)


@dataclass(frozen=True)
class KineticPoint:
    oxygen_energy: float
    steam_energy: float
    oxygen_order: float
    steam_order: float
    carbon_order: float
    gas_constant: float = 8.314462618

    def arguments(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class ObservedBoundary:
    pressure_external: Mapping[tuple[str, str], ObservedValue]
    pressure_barriers: Mapping[tuple[str, str], ObservedValue]
    sealed_pressure: frozenset[tuple[str, str]]
    heat_inward: Mapping[str, ObservedValue]
    radiation_ambient: Mapping[str, ObservedValue]
    sealed_heat: frozenset[str]

    def values(self):
        return BoundaryInputs(
            {key: item.value for key, item in self.pressure_external.items()},
            {key: Modulation(item.value) for key, item in self.pressure_barriers.items()},
            self.sealed_pressure,
            {key: item.value for key, item in self.heat_inward.items()},
            {key: item.value for key, item in self.radiation_ambient.items()}, self.sealed_heat,
        )


class SourceDerivativeFailure(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("fractional pressure derivative is unbounded at an active variable zero node")


class SparseWeakMaps:
    """Kronecker weak contractions and local face maps in C-order coordinates."""

    def __init__(self, projector):
        self.projector = projector
        nt, ns, nx = projector.state_shape
        ncell = ns*nx
        spatial = projector.spatial.reshape(projector.row_shape[1], ncell)
        cell = csr_matrix(spatial*np.asarray(projector.mesh.volume).ravel())
        integral = csr_matrix(projector.tests.integral)
        self.source = clean(kron(integral, cell, format="csr"))
        self.storage = clean(kron(-csr_matrix(projector.tests.derivative), cell, format="csr"))
        self.source_used = np.asarray(self.source.getnnz(axis=0)).ravel() > 0
        cells = np.arange(ncell).reshape(ns, nx)
        self.gradient, self.average, self.flux, self.trace, self.boundary = {}, {}, {}, {}, {}
        for axis, low, high, distance, area in (
                ("s", cells[:-1, :].ravel(), cells[1:, :].ravel(),
                 projector.mesh.distance_s, projector.mesh.area_s),
                ("x", cells[:, :-1].ravel(), cells[:, 1:].ravel(),
                 projector.mesh.distance_x, projector.mesh.area_x)):
            nface = len(low)
            rows = np.repeat(np.arange(nface), 2)
            columns = np.stack((low, high), axis=1).ravel()
            weights = np.stack((-1/np.asarray(distance).ravel(), 1/np.asarray(distance).ravel()), axis=1).ravel()
            gradient = coo_matrix((weights, (rows, columns)), shape=(nface, ncell)).tocsr()
            average = coo_matrix((np.full(2*nface, .5), (rows, columns)), shape=(nface, ncell)).tocsr()
            self.gradient[axis] = clean(kron(eye(nt, format="csr"), gradient, format="csr"))
            self.average[axis] = clean(kron(eye(nt, format="csr"), average, format="csr"))
            contraction = csr_matrix((spatial[:, high]-spatial[:, low])*np.asarray(area).ravel())
            self.flux[axis] = clean(kron(integral, contraction, format="csr"))
        for side, selection in (("inner", cells[:, 0]), ("outer", cells[:, -1]),
                                ("inlet", cells[0, :]), ("outlet", cells[-1, :])):
            indices = selection.ravel()
            trace = coo_matrix((np.ones(len(indices)), (np.arange(len(indices)), indices)),
                               shape=(len(indices), ncell)).tocsr()
            self.trace[side] = clean(kron(eye(nt, format="csr"), trace, format="csr"))
            contraction = csr_matrix(spatial[:, indices]*np.asarray(projector.mesh.boundary_areas[side]))
            self.boundary[side] = clean(kron(integral, contraction, format="csr"))


def _rate_derivatives(states, shapes, point, maps):
    temperature, carbon = states["T"].value.ravel(), states["carbon"].value.ravel()
    result, reports = [], {}
    for index, key, order, energy, pressure_scale in (
            (0, "pO2", point.oxygen_order, point.oxygen_energy, 2e4),
            (1, "pH2O", point.steam_order, point.steam_energy, 1e6)):
        pressure = states[key].value.ravel()
        shape = shapes[index].ravel()
        zero = pressure == 0
        variable = np.asarray(states[key].jacobian.getnnz(axis=1)).ravel() > 0
        active_zero = zero & variable & maps.source_used
        report = {
            "zero_nodes": int(np.count_nonzero(zero)),
            "variable_zero_nodes": int(np.count_nonzero(zero & variable)),
            "variable_zero_nodes_used_by_source": int(np.count_nonzero(active_zero)),
            "zero_nodes_outside_source_support": int(np.count_nonzero(zero & ~maps.source_used)),
            "zero_nodes_used_by_storage": int(np.count_nonzero(zero &
                (np.asarray(maps.storage.getnnz(axis=0)).ravel() > 0))),
            "order": float(order),
            "unbounded_active_derivative_nodes": int(np.count_nonzero(active_zero)) if order < 1 else 0,
            "raw_zero_observation_law_qualified": False if np.any(zero & variable) else None,
        }
        reports[key] = report
        if order < 1 and np.any(active_zero):
            raise SourceDerivativeFailure({"field": key, **report})
        slope = np.zeros_like(pressure)
        np.divide(order*shape, pressure, out=slope, where=pressure > 0)
        if order == 1:
            slope[zero] = np.exp(-energy/(point.gas_constant*temperature[zero]))*carbon[zero]**point.carbon_order/pressure_scale
        # For fractional zeros this finite placeholder is used ONLY after the
        # complete weak support or fixed-observation map removes their variation.
        derivative = scale_rows(states["T"].jacobian, shape*energy/(point.gas_constant*temperature**2))
        derivative += scale_rows(states["carbon"].jacobian, shape*point.carbon_order/carbon)
        derivative += scale_rows(states[key].jacobian, slope)
        result.append(clean(maps.source @ derivative))
    return result, reports


def linearize_balances(projector, layout, states, *, kinetics, relative_rate, transport,
                       gas_modulation, thermal_modulation, thermal_direction_ratio,
                       heat_knots, boundary, maps=None):
    """Return the full value assembly, shared sparse raw Jacobians and domain report.

    The observation maps are evaluated derivatives, including reconstruction if
    present. Geometry, time tests, kinetic parameters and transport ratios are
    fixed at this point. No finite differences, covariance weights or fitting.
    """
    if set(states) != set(FIELDS):
        raise ValueError("all five observed balance fields must be explicit")
    values = {key: finite(item.value, projector.state_shape, key) for key, item in states.items()}
    observed = list(states.values())+[gas_modulation, thermal_modulation]
    observed += list(boundary.pressure_external.values())+list(boundary.pressure_barriers.values())
    observed += list(boundary.heat_inward.values())+list(boundary.radiation_ambient.values())
    if any(item.jacobian.shape[1] != layout.size for item in observed):
        raise ValueError("every observed quantity must use the same raw coordinate layout")
    rates = arrhenius_pair(values, **kinetics.arguments())
    baseline = assemble_balances(projector, values, rates, relative_rate=relative_rate,
        transport=transport, gas_modulation=Modulation(gas_modulation.value),
        thermal_modulation=Modulation(thermal_modulation.value),
        thermal_direction_ratio=thermal_direction_ratio, heat_knots=heat_knots, boundary=boundary.values())
    maps = SparseWeakMaps(projector) if maps is None else maps
    if maps.projector is not projector:
        raise ValueError("sparse maps must belong to this exact projector")
    nrow, nraw = int(np.prod(projector.row_shape)), layout.size
    blocks = [[csr_matrix((nrow, nraw)) for _ in FIELDS] for _ in COLUMNS]

    def put(field, column, derivative):
        blocks[COLUMNS.index(column)][FIELDS.index(field)] += derivative

    rate_jacobian, domain = _rate_derivatives(states, rates.shapes, kinetics, maps)
    mixture = rate_jacobian[0]+relative_rate*rate_jacobian[1]
    put("carbon", "carbon_rate_scale", -mixture)
    put("pO2", "oxygen_response", -rate_jacobian[0])
    put("pH2O", "steam_response", -rate_jacobian[1])
    put("pProducts", "product_response", mixture)
    put("T", "heat_response", mixture)
    ratios = {"pO2": 1., "pH2O": transport.steam_ratio, "pProducts": transport.product_ratio}
    for key in PRESSURES:
        for axis in ("s", "x"):
            direction = transport.direction_ratio if axis == "s" else 1.
            gradient = maps.gradient[axis] @ values[key].ravel()
            modulation = maps.average[axis] @ gas_modulation.value.ravel()
            base = -ratios[key]*direction*modulation*gradient
            _, slope, _ = attenuated_flux(base, transport.forchheimer_product)
            derivative = scale_rows(maps.gradient[axis] @ states[key].jacobian,
                                    -ratios[key]*direction*modulation)
            derivative += scale_rows(maps.average[axis] @ gas_modulation.jacobian,
                                     -ratios[key]*direction*gradient)
            put(key, "gas_scale", maps.flux[axis] @ scale_rows(derivative, slope))
    for (key, side), external in boundary.pressure_external.items():
        trace = maps.trace[side] @ values[key].ravel()
        barrier = boundary.pressure_barriers[(key, side)]
        drive_jacobian = external.jacobian-maps.trace[side] @ states[key].jacobian
        derivative = scale_rows(drive_jacobian, barrier.value.ravel())
        derivative += scale_rows(barrier.jacobian, external.value.ravel()-trace)
        put(key, "surface_transfer", maps.boundary[side] @ derivative)
    for axis in ("s", "x"):
        direction = thermal_direction_ratio if axis == "s" else 1.
        gradient = maps.gradient[axis] @ values["T"].ravel()
        modulation = maps.average[axis] @ thermal_modulation.value.ravel()
        derivative = scale_rows(maps.gradient[axis] @ states["T"].jacobian, -direction*modulation)
        derivative += scale_rows(maps.average[axis] @ thermal_modulation.jacobian, -direction*gradient)
        put("T", "thermal_scale", maps.flux[axis] @ derivative)
    for side, item in boundary.heat_inward.items():
        put("T", "heat_input_scale", maps.boundary[side] @ item.jacobian)
    for side, ambient in boundary.radiation_ambient.items():
        trace = maps.trace[side] @ values["T"].ravel()
        derivative = scale_rows(maps.trace[side] @ states["T"].jacobian, -4*trace**3)
        derivative += scale_rows(ambient.jacobian, 4*ambient.value.ravel()**3)
        put("T", "radiation_scale", maps.boundary[side] @ derivative)
    _, fraction = heat_primitive(values["T"], heat_knots)
    put("T", "heat_capacity_contrast", -maps.storage @ scale_rows(states["T"].jacobian, fraction))
    target_jacobian = clean(vstack([maps.storage @ states[key].jacobian for key in FIELDS], format="csr"))
    column_jacobians = tuple(clean(vstack(parts, format="csr")) for parts in blocks)
    linearization = SharedObservationLinearization(target_jacobian, column_jacobians, layout.coordinate_keys)
    report = {"source_pressure_domains": domain,
              "zero_nodes_removed_from_observations": 0,
              "clipping_or_reconstruction_assumed": False,
              "observation_law_or_noise_calibration_qualified": False,
              "linearization_is_first_order_only": True}
    return baseline, linearization, report
