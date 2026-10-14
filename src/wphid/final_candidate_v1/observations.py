"""Adapt supplied observed fields only; never simulate or add random noise."""

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import csr_matrix

from wphid.weak.axisymmetric import axisymmetric_window_slices
from wphid.weak.axisymmetric_library import project_axisymmetric_pde_weak

from .problem import WeakCondition


def reaction_projection(t, z, r, window_shape, stride_shape):
    """Sparse tensor-Simpson operator with the historical radial volume weight."""
    coordinates = tuple(np.asarray(x, dtype=np.float64) for x in (t, z, r))
    if any(x.ndim != 1 or x.size < 3 or not np.isfinite(x).all() or
           not np.all(np.diff(x) > 0) for x in coordinates) or coordinates[2][0] < 0:
        raise ValueError("Invalid axisymmetric coordinates")
    shape = tuple(x.size for x in coordinates)
    index_grid = np.arange(np.prod(shape)).reshape(shape)
    indices, values, offsets = [], [], [0]
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        local = tuple(x[s] for x, s in zip(coordinates, slices))
        factors = []
        for x in local:
            u = (x - .5*(x[0]+x[-1])) / (.5*(x[-1]-x[0]))
            factors.append(simpson(np.eye(x.size), x=x, axis=1) * (1-u*u)**4)
        weights = np.einsum("i,j,k->ijk", *factors) * local[2][None, None, :]
        indices.extend(index_grid[slices].ravel())
        values.extend(weights.ravel())
        offsets.append(len(indices))
    if len(offsets) == 1:
        raise ValueError("No weak windows")
    return csr_matrix((np.asarray(values), np.asarray(indices), np.asarray(offsets)),
                      shape=(len(offsets)-1, int(np.prod(shape))))


def adapt_condition(*, name, t, z, r, observed_states, supplied_forcing,
                    window_shape=(15, 15, 15), stride_shape=(5, 5, 5)):
    """Three observed equations with mandatory axial/radial/self columns.

    Source terms are explicit inputs. No true parameters, support or partition
    enter this observation adapter. Positivity/clipping must be declared at
    acquisition; this function does not silently transform observed fields.
    """
    if len(observed_states) != 3 or len(supplied_forcing) != 3:
        raise ValueError("This adapter requires three states and source terms")
    bases, targets = [], []
    for state, forcing in zip(observed_states, supplied_forcing):
        base, target = project_axisymmetric_pde_weak(
            t=t, z=z, r=r, u=state, forcing=forcing,
            reaction_features=np.asarray(state)[..., None],
            window_shape=window_shape, stride_shape=stride_shape)
        bases.append(base)
        targets.append(target)
    projection = reaction_projection(t, z, r, window_shape, stride_shape)
    return WeakCondition(name, tuple(bases), tuple(targets),
                         tuple(np.asarray(s).ravel() for s in observed_states), projection)
