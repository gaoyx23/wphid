"""Fixed polynomial-modulated compact tests on unchanged observation nodes.

The original test is retained exactly. Three first-order temporal/axial/radial
modulations add weak balances, not samples or independent observations.
"""
import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss, legvander
from scipy.integrate import simpson
from scipy.sparse import csr_matrix, vstack
from wphid.weak.moment_corrected import MomentCorrectedAxisymmetricProjector, corrected_test_weights
from wphid.weak.axisymmetric import axisymmetric_window_slices
from wphid.weak.axisymmetric_library import _coordinate

MODES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def modulated_test_weights(x, kind, mode, *, degree=4):
    if mode == 0:
        return corrected_test_weights(x, kind, degree=degree)[0]
    x = _coordinate("x", x)
    if mode != 1 or kind not in {"mass", "first", "second", "radial_mass", "radial_diffusion"}:
        raise ValueError("only declared first-order compact modulations supported")
    if not isinstance(degree, int) or not 0 <= degree < len(x):
        raise ValueError("moment degree must be below node count")
    center, half = (x[0]+x[-1])/2, (x[-1]-x[0])/2
    polynomial = Legendre.basis(mode)
    def test(nodes):
        q = (nodes-center)/half
        b = 1-q*q
        p, dp, ddp = polynomial(q), polynomial.deriv()(q), polynomial.deriv(2)(q)
        value = b**4*p
        first = (-8*q*b**3*p+b**4*dp)/half
        second = (-8*b*b*(1-7*q*q)*p-16*q*b**3*dp+b**4*ddp)/half**2
        return {"mass": value, "first": first, "second": second,
                "radial_mass": nodes*value, "radial_diffusion": nodes*second+first}[kind]
    old = simpson(np.eye(len(x)), x=x, axis=0)*test(x)
    basis = legvander((x-center)/half, degree).T
    q, weights = leggauss(max(8, (degree+mode+11)//2))
    moments = legvander(q, degree).T@(half*weights*test(center+half*q))
    return old+np.linalg.lstsq(basis, moments-basis@old, rcond=None)[0]


class PolynomialTestProjector(MomentCorrectedAxisymmetricProjector):
    def __init__(self, *, t, z, r, window_shape, stride_shape, degree=4):
        super().__init__(t=t, z=z, r=r, window_shape=window_shape, stride_shape=stride_shape, degree=degree)
        self.test_modes = MODES
        self.original_window_count = self.operators["mass"].shape[0]
        indices = np.arange(np.prod(self.shape)).reshape(self.shape)
        all_operators = {name: [matrix] for name, matrix in self.operators.items()}
        for mode in MODES[1:]:
            values = {name: [] for name in all_operators}
            columns = []
            for block in axisymmetric_window_slices(self.shape, window_shape, stride_shape):
                tt, zz, rr = [nodes[sl] for nodes, sl in zip(self.coordinates, block)]
                tm = modulated_test_weights(tt, "mass", mode[0], degree=degree)
                td = modulated_test_weights(tt, "first", mode[0], degree=degree)
                zm = modulated_test_weights(zz, "mass", mode[1], degree=degree)
                zd = modulated_test_weights(zz, "second", mode[1], degree=degree)
                rm = modulated_test_weights(rr, "radial_mass", mode[2], degree=degree)
                rd = modulated_test_weights(rr, "radial_diffusion", mode[2], degree=degree)
                combinations = {"mass": (tm, zm, rm), "time": (td, zm, rm),
                                "axial": (tm, zd, rm), "radial": (tm, zm, rd)}
                columns.append(indices[block].ravel())
                for name, parts in combinations.items():
                    values[name].append(np.einsum("i,j,k->ijk", *parts).ravel())
            row_index = np.concatenate([np.full(len(c), i) for i, c in enumerate(columns)])
            column_index = np.concatenate(columns)
            for name in all_operators:
                all_operators[name].append(csr_matrix((np.concatenate(values[name]), (row_index, column_index)),
                    shape=(len(columns), int(np.prod(self.shape)))))
        self.operators = {name: vstack(matrices, format="csr") for name, matrices in all_operators.items()}
