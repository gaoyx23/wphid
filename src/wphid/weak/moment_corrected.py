"""Opt-in compact-test weak operators with low-order moment correction.

Only quadrature coefficients change. No field interpolation, differentiation,
truth parameter, new sensor or extra time point enters the operator.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.integrate import simpson
from scipy.sparse import csr_matrix

from wphid.weak.axisymmetric import axisymmetric_window_slices
from wphid.weak.axisymmetric_library import _bump_with_derivatives, _coordinate


def corrected_test_weights(x, kind, *, degree=4):
    """Minimum-L2 correction to Simpson test weights, exact through degree p.

    Among changes satisfying the moment constraints, this correction has the
    smallest Euclidean norm. That property does NOT imply minimum total noise
    variance or improved inference. Keep the uncorrected weights for comparison.
    """
    x = _coordinate("x", x)
    if not isinstance(degree, int) or not 0 <= degree < x.size:
        raise ValueError("degree must be an integer below the node count")
    if kind not in {"mass", "first", "second", "radial_mass", "radial_diffusion"}:
        raise ValueError("unknown test functional")
    center, half = (x[0] + x[-1])/2, (x[-1] - x[0])/2

    def test(nodes):
        q = (nodes - center)/half
        b = 1 - q*q
        value = b**4
        first = -8*q*b**3/half
        second = -8*b*b*(1 - 7*q*q)/half**2
        return {"mass": value, "first": first, "second": second,
                "radial_mass": nodes*value,
                "radial_diffusion": nodes*second + first}[kind]

    old = simpson(np.eye(x.size), x=x, axis=0) * test(x)
    basis = legvander((x - center)/half, degree).T
    # Test degree <= 9 plus basis degree p: Gaussian rule integrates exactly.
    q, gaussian = leggauss(max(8, (degree + 11)//2))
    nodes = center + half*q
    moments = legvander(q, degree).T @ (half*gaussian*test(nodes))
    correction = np.linalg.lstsq(basis, moments - basis @ old, rcond=None)[0]
    corrected = old + correction
    return corrected, old


class MomentCorrectedAxisymmetricProjector:
    """Same weak PDE equation as the legacy projector; opt-in new quadrature."""

    def __init__(self, *, t, z, r, window_shape, stride_shape, degree=4):
        self.coordinates = tuple(_coordinate(name, nodes) for name, nodes in
                                 (("t", t), ("z", z), ("r", r)))
        self.shape = tuple(len(nodes) for nodes in self.coordinates)
        self.degree = degree
        indices = np.arange(np.prod(self.shape)).reshape(self.shape)
        operators = {name: [] for name in ("mass", "time", "axial", "radial")}
        old_norms = {name: [] for name in operators}
        columns = []
        for block in axisymmetric_window_slices(self.shape, window_shape, stride_shape):
            tt, zz, rr = [nodes[sl] for nodes, sl in zip(self.coordinates, block)]
            tm, tm0 = corrected_test_weights(tt, "mass", degree=degree)
            td, td0 = corrected_test_weights(tt, "first", degree=degree)
            zm, zm0 = corrected_test_weights(zz, "mass", degree=degree)
            zd, zd0 = corrected_test_weights(zz, "second", degree=degree)
            rm, rm0 = corrected_test_weights(rr, "radial_mass", degree=degree)
            rd, rd0 = corrected_test_weights(rr, "radial_diffusion", degree=degree)
            combinations = {"mass": (tm, zm, rm), "time": (td, zm, rm),
                            "axial": (tm, zd, rm), "radial": (tm, zm, rd)}
            old = {"mass": (tm0, zm0, rm0), "time": (td0, zm0, rm0),
                   "axial": (tm0, zd0, rm0), "radial": (tm0, zm0, rd0)}
            columns.append(indices[block].ravel())
            for name, parts in combinations.items():
                operators[name].append(np.einsum("i,j,k->ijk", *parts).ravel())
                old_norms[name].append(float(np.prod([np.linalg.norm(p) for p in old[name]])))
        if not columns:
            raise ValueError("weak window selection must be nonempty")
        row_index = np.concatenate([np.full(len(c), i) for i, c in enumerate(columns)])
        column_index = np.concatenate(columns)
        self.operators = {name: csr_matrix((np.concatenate(values), (row_index, column_index)),
                                           shape=(len(columns), int(np.prod(self.shape))))
                          for name, values in operators.items()}
        self.weight_diagnostics = {}
        for name, values in operators.items():
            corrected = np.asarray([np.linalg.norm(v) for v in values])
            original = np.asarray(old_norms[name])
            self.weight_diagnostics[name] = {
                "minimum_row_norm_ratio": float(np.min(corrected/original)),
                "maximum_row_norm_ratio": float(np.max(corrected/original)),
                "interpretation": "conditional IID nodal-noise sensitivity, not a fitted-parameter uncertainty bound"}

    def _values(self, field):
        values = np.asarray(field, dtype=float)
        if values.shape != self.shape or not np.all(np.isfinite(values)):
            raise ValueError("finite field must match the projector grid")
        return values.ravel()

    def mass(self, field):
        return np.asarray(self.operators["mass"] @ self._values(field))

    def library(self, u, forcing):
        values = self._values(u)
        matrix = np.column_stack(tuple(self.operators[name] @ values
                                       for name in ("axial", "radial", "mass")))
        target = -self.operators["time"] @ values - self.mass(forcing)
        return matrix, np.asarray(target)
