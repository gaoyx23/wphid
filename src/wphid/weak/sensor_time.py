"""Temporal weak ODE balances at observed locations, without spatial mixing.

Only local evolution equations are supported. This operator cannot replace a
spatial PDE operator for transport or diffusion. Every location is retained;
there is no truth-based activation selection or spatial quadrature weighting.
"""
import numpy as np
from scipy.integrate import simpson


def project_sensor_time_weak(*, t, u, features, window, stride):
    time = np.asarray(t, dtype=float)
    values = np.asarray(u, dtype=float)
    candidates = np.asarray(features, dtype=float)
    if (time.ndim != 1 or time.size < 3 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError("finite strictly increasing time required")
    if (values.ndim < 2 or values.shape[0] != time.size
            or candidates.ndim != values.ndim+1 or candidates.shape[:-1] != values.shape
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(candidates))):
        raise ValueError("finite time-by-location values and aligned feature arrays required")
    if (not isinstance(window, int) or not isinstance(stride, int)
            or window < 3 or window > time.size or stride < 1):
        raise ValueError("valid integer temporal window and stride required")
    rows, targets = [], []
    for start in range(0, time.size-window+1, stride):
        tt = time[start:start+window]
        half = (tt[-1]-tt[0])/2
        q = (tt-(tt[-1]+tt[0])/2)/half
        base = 1-q*q
        phi, derivative = base**4, -8*q*base**3/half
        reshape = (-1,)+(1,)*(values.ndim-1)
        local = values[start:start+window]
        # A constant offset has zero continuum weak derivative. Subtract it to
        # avoid subtracting large state baselines when degradation is very small.
        target = -simpson((local-local[0])*derivative.reshape(reshape), x=tt, axis=0)
        matrix = simpson(candidates[start:start+window]*phi.reshape(reshape+(1,)), x=tt, axis=0)
        rows.append(matrix.reshape(-1,candidates.shape[-1]))
        targets.append(target.reshape(-1))
    return np.vstack(rows), np.concatenate(targets)
