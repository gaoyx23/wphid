"""Conservative face contractions and unknown heat-capacity storage.

Volumes and face measures use one consistent angular convention. Supplied mesh,
tests, fields and boundary inputs are declared model inputs, never inferred truth.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np


SIDES = ("inner", "outer", "inlet", "outlet")


def finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class Modulation:
    value: np.ndarray
    jacobian: Mapping[str, np.ndarray] = field(default_factory=dict)

    def checked(self, shape, name):
        value = finite(self.value, shape, name)
        if np.any(value < 0):
            raise ValueError(f"{name} cannot be negative")
        derivatives = {}
        for key, derivative in self.jacobian.items():
            if not isinstance(key, str) or not key:
                raise ValueError("modulation derivative names must be nonempty strings")
            derivatives[key] = finite(derivative, shape, name+" derivative")
        return value, derivatives


@dataclass(frozen=True)
class ShellMesh:
    volume: np.ndarray
    area_s: np.ndarray
    area_x: np.ndarray
    distance_s: np.ndarray
    distance_x: np.ndarray
    boundary_areas: Mapping[str, np.ndarray]

    def __post_init__(self):
        volume = np.asarray(self.volume, dtype=float)
        if volume.ndim != 2 or min(volume.shape) < 1:
            raise ValueError("volume must be a nonempty two-dimensional cell measure")
        ns, nx = volume.shape
        checks = ((volume, (ns, nx), "volume"),
                  (self.area_s, (ns-1, nx), "area_s"),
                  (self.area_x, (ns, nx-1), "area_x"),
                  (self.distance_s, (ns-1, nx), "distance_s"),
                  (self.distance_x, (ns, nx-1), "distance_x"))
        for value, shape, name in checks:
            array = finite(value, shape, name)
            if np.any(array <= 0):
                raise ValueError(f"{name} must be positive")
        if set(self.boundary_areas) != set(SIDES):
            raise ValueError("all four boundary measures must be explicit")
        for side in SIDES:
            size = ns if side in ("inner", "outer") else nx
            if np.any(finite(self.boundary_areas[side], (size,), side+" area") <= 0):
                raise ValueError("boundary measures must be positive")

    @property
    def shape(self):
        return np.asarray(self.volume).shape


@dataclass(frozen=True)
class TemporalTests:
    integral: np.ndarray
    derivative: np.ndarray

    def __post_init__(self):
        weights = np.asarray(self.integral, dtype=float)
        if weights.ndim != 2 or min(weights.shape) < 1:
            raise ValueError("temporal weights must be nonempty windows by time")
        finite(weights, weights.shape, "temporal integral")
        finite(self.derivative, weights.shape, "temporal derivative")


def temporal_tests(times, *, window_samples, stride_samples):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 3 or not np.all(np.isfinite(times)):
        raise ValueError("finite time coordinates are required")
    delta = np.diff(times)
    if np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-10, atol=1e-10):
        raise ValueError("this Simpson implementation requires a uniform increasing grid")
    if not isinstance(window_samples, int) or window_samples < 3 or window_samples % 2 != 1:
        raise ValueError("window_samples must be an odd integer of at least three")
    if not isinstance(stride_samples, int) or stride_samples < 1:
        raise ValueError("stride_samples must be a positive integer")
    starts = tuple(range(0, len(times)-window_samples+1, stride_samples))
    if not starts:
        raise ValueError("time sequence is shorter than the declared window")
    integral = np.zeros((len(starts), len(times)))
    derivative = np.zeros_like(integral)
    for row, start in enumerate(starts):
        selection = slice(start, start+window_samples)
        local = times[selection]
        coordinate = 2*(local-local[0])/(local[-1]-local[0])-1
        quadrature = np.ones(window_samples)
        quadrature[1:-1:2], quadrature[2:-1:2] = 4., 2.
        quadrature *= delta[0]/3
        integral[row, selection] = quadrature*(1-coordinate**2)
        derivative[row, selection] = quadrature*(-4*coordinate/(local[-1]-local[0]))
    return TemporalTests(integral, derivative)


def heat_primitive(temperature, knots):
    """Return I(T), where I'=clip((T-T0)/(T1-T0),0,1), with I(T0)=0."""
    temperature = np.asarray(temperature, dtype=float)
    if not np.all(np.isfinite(temperature)):
        raise ValueError("temperature must be finite")
    lower, upper = map(float, knots)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("heat-capacity knots must be finite and ordered")
    offset = np.clip(temperature-lower, 0., upper-lower)
    primitive = offset**2/(2*(upper-lower)) + np.maximum(temperature-upper, 0.)
    fraction = np.clip((temperature-lower)/(upper-lower), 0., 1.)
    return primitive, fraction


def attenuated_flux(base_flux, product):
    """Flux, derivative in base flux, and derivative in alpha*reference scale.

    The scale multiplying this shape remains linear. The physical alpha is
    separable from its product only when a nonzero reference scale is identified.
    """
    value = np.asarray(base_flux, dtype=float)
    if not np.all(np.isfinite(value)) or not np.isfinite(product) or product < 0:
        raise ValueError("finite flux and a nonnegative Forchheimer product are required")
    root = np.sqrt(1+4*product*np.abs(value))
    flux = 2*value/(1+root)
    return flux, 1/root, -np.sign(value)*flux**2/root


class WeakProjector:
    """Space-time weak contractions of cell storage, sources and face fluxes."""

    def __init__(self, mesh: ShellMesh, tests: TemporalTests, spatial_tests):
        self.mesh, self.tests = mesh, tests
        spatial = np.asarray(spatial_tests, dtype=float)
        if spatial.ndim != 3 or spatial.shape[0] < 1:
            raise ValueError("spatial tests must be tests by meridional by thickness cells")
        self.spatial = finite(spatial, (spatial.shape[0], *mesh.shape), "spatial tests")
        self.nt = tests.integral.shape[1]
        self.state_shape = (self.nt, *mesh.shape)
        self.row_shape = (tests.integral.shape[0], spatial.shape[0])

    def cell_integrals(self, values):
        values = finite(values, self.state_shape, "cell field")
        return np.einsum("tij,mij,ij->tm", values, self.spatial, self.mesh.volume)

    def source(self, values):
        return (self.tests.integral @ self.cell_integrals(values)).ravel()

    def storage(self, values):
        return (-self.tests.derivative @ self.cell_integrals(values)).ravel()

    def flux(self, flux_s, flux_x):
        ns, nx = self.mesh.shape
        flux_s = finite(flux_s, (self.nt, ns-1, nx), "meridional face flux")
        flux_x = finite(flux_x, (self.nt, ns, nx-1), "thickness face flux")
        delta_s = self.spatial[:, 1:, :] - self.spatial[:, :-1, :]
        delta_x = self.spatial[:, :, 1:] - self.spatial[:, :, :-1]
        spatial = np.einsum("tij,mij,ij->tm", flux_s, delta_s, self.mesh.area_s)
        spatial += np.einsum("tij,mij,ij->tm", flux_x, delta_x, self.mesh.area_x)
        return (self.tests.integral @ spatial).ravel()

    def trace(self, values, side):
        values = finite(values, self.state_shape, "boundary cell field")
        if side == "inner":
            return values[:, :, 0]
        if side == "outer":
            return values[:, :, -1]
        if side == "inlet":
            return values[:, 0, :]
        if side == "outlet":
            return values[:, -1, :]
        raise ValueError("unknown boundary side")

    def boundary(self, inward_flux, side):
        if side not in SIDES:
            raise ValueError("unknown boundary side")
        # Extract every spatial test on the same face, independent of time.
        if side in ("inner", "outer"):
            phi = self.spatial[:, :, 0 if side == "inner" else -1]
        else:
            phi = self.spatial[:, 0 if side == "inlet" else -1, :]
        size = self.mesh.shape[0] if side in ("inner", "outer") else self.mesh.shape[1]
        shape = (self.nt, size)
        inward = finite(inward_flux, shape, "inward boundary flux")
        spatial = (inward*self.mesh.boundary_areas[side]) @ phi.T
        return (self.tests.integral @ spatial).ravel()

    def gradient(self, values):
        values = finite(values, self.state_shape, "gradient field")
        return (np.diff(values, axis=1)/self.mesh.distance_s,
                np.diff(values, axis=2)/self.mesh.distance_x)

    def face_modulation(self, values):
        values = finite(values, self.state_shape, "modulation")
        return (.5*(values[:, 1:, :]+values[:, :-1, :]),
                .5*(values[:, :, 1:]+values[:, :, :-1]))
