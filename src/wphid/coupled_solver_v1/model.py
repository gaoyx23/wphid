"""Value/parameter assembler, independent of observation derivative evaluation.

All physical coefficients and constitutive coordinates are supplied/estimated.
This reuses the frozen conservative nuisance assembler. A separate qualified
raw-coordinate path is used only for uncertainty diagnostics. No truth import.
"""

from dataclasses import dataclass

import numpy as np

from wphid.coupled_balance_v1.assembler import (
    BoundaryInputs, COLUMNS, FIELDS, RatePair, TransportShape, WeakSystem,
    assemble_balances,
)
from wphid.coupled_balance_v1.operators import Modulation
from wphid.coupled_candidates_v1 import gas_modulation, thermal_modulation
from wphid.coupled_observation_v1 import ObservedValue


@dataclass(frozen=True)
class Candidate:
    source: str
    gas: str = "combined_opening"
    thermal: str = "stiffness"
    flux: str = "Darcy"

    @property
    def identity(self):
        return "/".join((self.source, self.gas, self.thermal, self.flux))


def candidate_space():
    """Twelve declared models; not the Cartesian product of all assumptions."""
    sources = ("null", "oxygen_only", "steam_only", "both_shared", "both_free",
               "both_swapped_product", "both_oxygen_heat", "both_independent")
    return tuple(Candidate(s) for s in sources) + (
        Candidate("both_shared", gas="constant"),
        Candidate("both_shared", gas="crack_only"),
        Candidate("both_shared", thermal="constant"),
        Candidate("both_shared", flux="Forchheimer"),
    )


def active_paths(source):
    if source == "null":
        return ()
    if source == "oxygen_only":
        return ("oxygen",)
    if source == "steam_only":
        return ("steam",)
    return ("oxygen", "steam")


def source_terms(source):
    paths = active_paths(source)
    if source in ("both_free", "both_independent"):
        return tuple((field, path, f"{field}.{path}") for path in paths
            for field in ("carbon", "pO2" if path == "oxygen" else "pH2O", "pProducts", "T"))
    return ()


def parameter_box(candidate):
    box = {}
    if candidate.source == "both_independent":
        for _, _, term in source_terms(candidate.source):
            box[f"kinetic.{term}.energy_1e5"] = (.6, 1.8)
            box[f"kinetic.{term}.pressure_order"] = (.3, 1.8)
            box[f"kinetic.{term}.carbon_order"] = (.2, 2.5)
    else:
        for path in active_paths(candidate.source):
            box[f"kinetic.{path}.energy_1e5"] = (.6, 1.8)
            box[f"kinetic.{path}.pressure_order"] = (.3, 1.8)
        if active_paths(candidate.source):
            box["kinetic.carbon_order"] = (.2, 2.5)
        if len(active_paths(candidate.source)) == 2 and candidate.source != "both_free":
            box["reaction.relative_rate"] = (.3, 8.)
    box.update({"gas.direction_ratio": (.3, 3.), "gas.steam_ratio": (.3, 3.),
                "gas.product_ratio": (.3, 3.), "thermal.direction_ratio": (.5, 4.),
                "boundary.barrier_strength": (0., 2.)})
    if candidate.flux == "Forchheimer":
        box["gas.forchheimer_product"] = (1e-10, 2e-6)
    if candidate.gas == "combined_opening":
        box.update({"gas.modulation.stress_gain_per_pa": (1e-9, 1e-6),
            "gas.modulation.stress_cap": (.1, 10.), "gas.modulation.damage_gain": (.1, 15.),
            "gas.modulation.carbon_gain": (.1, 15.), "gas.modulation.closure_gain": (0., 2.),
            "gas.modulation.closure_temperature": (1200., 1800.),
            "gas.modulation.closure_span": (200., 700.), "gas.modulation.total_cap": (5., 25.)})
    elif candidate.gas == "crack_only":
        box["gas.modulation.gain"] = (.1, 15.)
    if candidate.thermal == "stiffness":
        box.update({"thermal.modulation.loss": (.1, .7),
            "thermal.modulation.stiffness_floor": (.05, .2),
            "thermal.modulation.conductivity_floor": (.2, .5)})
    return box


def rate_expression(values, path, point, prefix, carbon_name):
    pressure_key, reference = ("pO2", 2e4) if path == "oxygen" else ("pH2O", 1e6)
    T, C, P = values["T"], values["carbon"], values[pressure_key]
    if np.any(T <= 0) or np.any(C <= 0) or np.any(P < 0):
        raise ValueError("kinetic domain failure, no observation clipping")
    E, m, q = point[prefix+"energy_1e5"], point[prefix+"pressure_order"], point[carbon_name]
    phi = np.exp(-1e5*E/(8.314462618*T))*(P/reference)**m*C**q
    logP = np.zeros_like(P)
    np.log(P/reference, out=logP, where=P > 0)
    return phi, {prefix+"energy_1e5": -1e5*phi/(8.314462618*T),
                 prefix+"pressure_order": phi*logP, carbon_name: phi*np.log(C)}


class ValueProblem:
    def __init__(self, projector, fields, boundary, *, heat_knots=(293.15, 1873.15)):
        self.projector, self.fields, self.boundary, self.heat_knots = projector, fields, boundary, heat_knots
        self.drivers = {key: ObservedValue.fixed(fields[key], 1) for key in ("T", "carbon", "crack", "stress")}
        self.target = np.concatenate([projector.storage(fields[key]) for key in FIELDS])
        self.block_rows = int(np.prod(projector.row_shape))

    def assemble(self, candidate, point):
        required = parameter_box(candidate)
        if set(point) != set(required) or any(not low <= point[k] <= high for k, (low, high) in required.items()):
            raise ValueError("all and only declared outer coordinates are required")
        gaspoint = {k.removeprefix("gas.modulation."): v for k, v in point.items() if k.startswith("gas.modulation.")}
        thermalpoint = {k.removeprefix("thermal.modulation."): v for k, v in point.items() if k.startswith("thermal.modulation.")}
        gas = gas_modulation(candidate.gas, self.drivers, gaspoint)
        thermal = thermal_modulation(candidate.thermal, self.drivers, thermalpoint)
        barriers = {}
        for key in self.boundary.pressure_external:
            coverage = self.projector.trace(self.fields["crack"], key[1]) if key[1] == "inner" else np.zeros_like(self.boundary.pressure_external[key])
            value = np.exp(-point["boundary.barrier_strength"]*coverage)
            barriers[key] = Modulation(value, {"boundary.barrier_strength": -coverage*value})
        boundary = BoundaryInputs(self.boundary.pressure_external, barriers, self.boundary.sealed_pressure,
            self.boundary.heat_inward, self.boundary.radiation_ambient, self.boundary.sealed_heat)
        base = assemble_balances(self.projector, {k: self.fields[k] for k in FIELDS},
            RatePair(np.zeros((2, *self.projector.state_shape))), relative_rate=1.,
            transport=TransportShape(point["gas.direction_ratio"], point["gas.steam_ratio"], point["gas.product_ratio"], point.get("gas.forchheimer_product", 0.)),
            gas_modulation=Modulation(gas.value, gas.parameters),
            thermal_modulation=Modulation(thermal.value, thermal.parameters),
            thermal_direction_ratio=point["thermal.direction_ratio"], heat_knots=self.heat_knots, boundary=boundary)
        sources = []
        if candidate.source in ("both_free", "both_independent"):
            for field, path, term in source_terms(candidate.source):
                prefix = f"kinetic.{term}." if candidate.source == "both_independent" else f"kinetic.{path}."
                qname = prefix+"carbon_order" if candidate.source == "both_independent" else "kinetic.carbon_order"
                phi, derivative = rate_expression(self.fields, path, point, prefix, qname)
                sign = -1 if field in ("carbon", "pO2", "pH2O") else 1
                sources.append((field, term+"_response", sign*phi, {k: sign*v for k,v in derivative.items()}))
        elif active_paths(candidate.source):
            rates = {p: rate_expression(self.fields, p, point, f"kinetic.{p}.", "kinetic.carbon_order") for p in active_paths(candidate.source)}
            if len(rates) == 1:
                mixture, md = next(iter(rates.values()))
            else:
                o, od = rates["oxygen"]; w, wd = rates["steam"]; ratio = point["reaction.relative_rate"]
                mixture = o+ratio*w
                md = {k: od.get(k, 0)+ratio*wd.get(k, 0) for k in od.keys() | wd.keys()}
                md["reaction.relative_rate"] = w
            sources.append(("carbon", "carbon_rate_scale", -mixture, {k:-v for k,v in md.items()}))
            for path, field in (("oxygen", "pO2"), ("steam", "pH2O")):
                if path in rates:
                    phi, derivative = rates[path]
                    sources.append((field, path+"_response", -phi, {k:-v for k,v in derivative.items()}))
            product, pd, heat, hd = mixture, md, mixture, md
            if candidate.source == "both_swapped_product":
                product = ratio*o+w
                pd = {k:ratio*od.get(k,0)+wd.get(k,0) for k in od.keys() | wd.keys()}
                pd["reaction.relative_rate"] = o
            if candidate.source == "both_oxygen_heat":
                heat, hd = rates["oxygen"]
            sources.extend((("pProducts", "product_response", product, pd), ("T", "heat_response", heat, hd)))
        keys = tuple(s[1] for s in sources)+COLUMNS[5:]
        G = np.zeros((len(base.target), len(keys)))
        G[:,len(sources):] = base.matrix[:,5:]
        derivatives = {k: np.column_stack((np.zeros((len(base.target),len(sources))),v[:,5:])) for k,v in base.matrix_jacobian.items() if k in point}
        for j,(field, _, phi, derivative) in enumerate(sources):
            rows = slice(FIELDS.index(field)*self.block_rows, (FIELDS.index(field)+1)*self.block_rows)
            G[rows,j] = self.projector.source(phi)
            for k,v in derivative.items():
                derivatives.setdefault(k,np.zeros_like(G))[rows,j] = self.projector.source(v)
        return WeakSystem(base.target, G, derivatives, base.row_keys, keys)
