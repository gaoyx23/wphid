"""Interpolation with explicit measured-station provenance; not state inference.

No temporal filtering, boundary extrapolation, or hidden field access. Preserves
input station values; between-station fields remain reconstructed, not measured.
"""
import numpy as np
from scipy.interpolate import interp1d,PchipInterpolator


def reconstruct_observed_stations(*,stations,values,target_stations,method):
    s=np.asarray(stations,dtype=float);target=np.asarray(target_stations,dtype=float)
    v=np.asarray(values,dtype=float)
    if (s.ndim!=1 or len(s)<2 or target.ndim!=1 or not len(target)
        or np.any(np.diff(s)<=0) or np.any(np.diff(target)<=0)
        or not np.all(np.isfinite(s)) or not np.all(np.isfinite(target))
        or v.ndim!=3 or v.shape[1]!=len(s) or not np.all(np.isfinite(v))):
        raise ValueError("finite time-by-station-by-depth values and increasing stations required")
    if target[0]<s[0] or target[-1]>s[-1]:raise ValueError("spatial extrapolation prohibited")
    if method=="linear":out=interp1d(s,v,axis=1,bounds_error=True)(target)
    elif method=="pchip":out=PchipInterpolator(s,v,axis=1,extrapolate=False)(target)
    else:raise ValueError("only declared linear/pchip interpolation supported")
    # Preserve supplied measurements exactly where target coordinates coincide.
    for i,coordinate in enumerate(s):
        hits=np.flatnonzero(target==coordinate)
        if len(hits):out[:,hits[0],:]=v[:,i,:]
    return out
