"""Unknown constitutive combinations with raw and parameter derivatives.

No truth module or default material value is imported. Every numerical point is
supplied. Product coordinates remove an opening/amplitude gauge; they do not
separate the original gain, stress threshold or open/closed diffusivity ratio.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from wphid.coupled_observation_v1.linearization import ObservedValue, clean, scale_rows


class ModulationKink(ValueError):
    pass


@dataclass(frozen=True)
class Expression:
    value: np.ndarray
    jacobian: object
    parameters: dict

    @classmethod
    def observed(cls, item):
        return cls(item.value, item.jacobian, {})

    @classmethod
    def constant(cls, value, reference, *, parameter=None):
        array = np.full(reference.value.shape, value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("finite constitutive point required")
        derivatives = {} if parameter is None else {parameter: np.ones_like(array)}
        return cls(array, csr_matrix(reference.jacobian.shape), derivatives)

    def observed_value(self):
        return ObservedValue(self.value, self.jacobian)

    def __add__(self, other):
        other = other if isinstance(other, Expression) else self.constant(other, self)
        parameters = {key: self.parameters.get(key, 0)+other.parameters.get(key, 0)
                      for key in self.parameters.keys() | other.parameters.keys()}
        return Expression(self.value+other.value, clean(self.jacobian+other.jacobian), parameters)

    __radd__ = __add__

    def __neg__(self):
        return Expression(-self.value, -self.jacobian, {key: -value for key, value in self.parameters.items()})

    def __sub__(self, other):
        return self+-other if isinstance(other, Expression) else self+-float(other)

    def __rsub__(self, other):
        return -self+other

    def __mul__(self, other):
        other = other if isinstance(other, Expression) else self.constant(other, self)
        parameters = {key: self.parameters.get(key, 0)*other.value+other.parameters.get(key, 0)*self.value
                      for key in self.parameters.keys() | other.parameters.keys()}
        return Expression(self.value*other.value,
            clean(scale_rows(self.jacobian, other.value)+scale_rows(other.jacobian, self.value)), parameters)

    __rmul__ = __mul__

    def reciprocal(self):
        if np.any(self.value == 0):
            raise ValueError("constitutive division by zero")
        slope = -1/self.value**2
        return Expression(1/self.value, scale_rows(self.jacobian, slope),
                          {key: value*slope for key, value in self.parameters.items()})

    def __truediv__(self, other):
        other = other if isinstance(other, Expression) else self.constant(other, self)
        return self*other.reciprocal()


def _clip(expression, lower, upper=None, *, lower_is_physical_domain=False):
    lower = lower if isinstance(lower, Expression) else Expression.constant(lower, expression)
    upper = upper if isinstance(upper, Expression) or upper is None else Expression.constant(upper, expression)
    if upper is not None and np.any(lower.value >= upper.value):
        raise ValueError("ordered clipping bounds required")
    below = expression.value < lower.value
    above = np.zeros_like(below) if upper is None else expression.value > upper.value
    for edge, bound, permitted in ((expression.value == lower.value, lower, lower_is_physical_domain),
            (np.zeros_like(below) if upper is None else expression.value == upper.value, upper, False)):
        if bound is None or permitted or not np.any(edge):
            continue
        raw_difference = clean(expression.jacobian-bound.jacobian)
        changed = np.asarray(raw_difference.getnnz(axis=1)).ravel().reshape(expression.value.shape) > 0
        for key in expression.parameters.keys() | bound.parameters.keys():
            changed |= np.broadcast_to(expression.parameters.get(key, 0)-bound.parameters.get(key, 0), changed.shape) != 0
        if np.any(edge & changed):
            raise ModulationKink("active constitutive clipping kink; no silent zero derivative")
    inside = ~below & ~above
    value = np.where(below, lower.value, expression.value)
    jacobian = scale_rows(expression.jacobian, inside)+scale_rows(lower.jacobian, below)
    names = expression.parameters.keys() | lower.parameters.keys()
    if upper is not None:
        value = np.where(above, upper.value, value)
        jacobian += scale_rows(upper.jacobian, above)
        names |= upper.parameters.keys()
    parameters = {key: np.broadcast_to(expression.parameters.get(key, 0), value.shape)*inside+
        np.broadcast_to(lower.parameters.get(key, 0), value.shape)*below+
        (0 if upper is None else np.broadcast_to(upper.parameters.get(key, 0), value.shape)*above) for key in names}
    return Expression(value, clean(jacobian), parameters)


def _inputs(fields):
    required = ("T", "carbon", "crack", "stress")
    if not set(required).issubset(fields):
        raise ValueError("temperature, carbon, crack and stress observations must be explicit")
    values = {key: Expression.observed(fields[key]) for key in required}
    shape, width = values["T"].value.shape, values["T"].jacobian.shape[1]
    if any(item.value.shape != shape or item.jacobian.shape[1] != width for item in values.values()):
        raise ValueError("physical drivers must share shape and raw coordinates")
    if (np.any(values["T"].value <= 0) or np.any(values["carbon"].value < 0) or
            np.any(values["carbon"].value > 1) or np.any(values["crack"].value < 0) or
            np.any(values["crack"].value > 1) or np.any(values["stress"].value < 0)):
        raise ValueError("physical drivers outside declared domains")
    return values


def _point(point, names, reference, prefix):
    if set(point) != set(names):
        raise ValueError("exactly the declared unknown constitutive point is required")
    if any(not np.isfinite(value) or value < 0 for value in point.values()):
        raise ValueError("finite nonnegative constitutive coordinates required")
    return {key: Expression.constant(point[key], reference, parameter=prefix+key) for key in names}


def gas_modulation(kind, fields, point):
    drivers = _inputs(fields)
    reference = drivers["T"]
    names = {"constant": (), "crack_only": ("gain",), "carbon_loss_only": ("gain",),
             "stress_only": ("stress_gain_per_pa", "stress_cap"),
             "combined_opening": ("stress_gain_per_pa", "stress_cap", "damage_gain", "carbon_gain",
                                  "closure_gain", "closure_temperature", "closure_span", "total_cap")}
    if kind not in names:
        raise ValueError("unsupported declared gas modulation")
    p = _point(point, names[kind], reference, "gas.modulation.")
    if kind == "constant":
        return Expression.constant(1., reference)
    if kind == "crack_only":
        return 1+p["gain"]*drivers["crack"]
    if kind == "carbon_loss_only":
        return 1+p["gain"]*(1-drivers["carbon"])
    if point["stress_cap"] <= 0:
        raise ValueError("positive stress saturation required")
    stress = _clip(p["stress_gain_per_pa"]*drivers["stress"], 0., p["stress_cap"], lower_is_physical_domain=True)
    if kind == "stress_only":
        return 1+stress
    if point["closure_span"] <= 0 or point["total_cap"] <= 0:
        raise ValueError("positive thermal span and opening saturation required")
    hot = _clip((drivers["T"]-p["closure_temperature"])/p["closure_span"], 0., 1.)
    opening = stress+p["damage_gain"]*drivers["crack"]+p["carbon_gain"]*(1-drivers["carbon"])-p["closure_gain"]*hot
    return 1+_clip(opening, 0., p["total_cap"])


def thermal_modulation(kind, fields, point):
    drivers = _inputs(fields)
    reference = drivers["T"]
    names = {"constant": (), "stiffness": ("loss", "stiffness_floor", "conductivity_floor")}
    if kind not in names:
        raise ValueError("unsupported declared thermal modulation")
    p = _point(point, names[kind], reference, "thermal.modulation.")
    if kind == "constant":
        return Expression.constant(1., reference)
    if point["loss"] >= 1 or not 0 < point["conductivity_floor"] < 1 or not 0 <= point["stiffness_floor"] < 1:
        raise ValueError("thermal shape outside its declared physical domain")
    # carbon*(1-crack)<=1 on the declared driver domain, so the archived upper
    # stiffness cap is redundant. Its physical-boundary tangent is retained.
    stiffness = _clip(drivers["carbon"]*(1-drivers["crack"]), p["stiffness_floor"])
    return _clip(1-p["loss"]*(1-stiffness), p["conductivity_floor"])


def exponential_barrier(coverage, strength):
    if not np.isfinite(strength) or strength < 0 or np.any(coverage.value < 0) or np.any(coverage.value > 1):
        raise ValueError("declared coverage in [0,1] and nonnegative barrier strength required")
    value = np.exp(-strength*coverage.value)
    return Expression(value, scale_rows(coverage.jacobian, -strength*value),
                      {"boundary.barrier_strength": -coverage.value*value})
