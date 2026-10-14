"""One fixed noise-scaled pressure interpolation proposal, not a PDE solution.

asinh(p/sigma) is defined at zero, locally linear near zero and logarithmic at
large positive pressure. Sigma is declared raw measurement scale, not fitted.
The proposal supplies no calibrated error bound between sparse stations.
"""
import numpy as np
from .observed_spatial_reconstruction import reconstruct_observed_stations


def reconstruct_pressure_scaled(*,stations,values,target_stations,noise_sigma):
    sigma=float(noise_sigma)
    if not np.isfinite(sigma) or sigma<=0:raise ValueError("declared positive finite measurement sigma required")
    values=np.asarray(values,dtype=float)
    transformed=np.arcsinh(values/sigma)
    reconstructed=reconstruct_observed_stations(stations=stations,values=transformed,
        target_stations=target_stations,method="linear")
    result=sigma*np.sinh(reconstructed)
    for i,station in enumerate(np.asarray(stations)):
        hits=np.flatnonzero(np.asarray(target_stations)==station)
        if len(hits):result[:,hits[0],:]=values[:,i,:]
    if not np.all(np.isfinite(result)):raise ValueError("nonfinite decoded pressure")
    return result
