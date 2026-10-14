"""Depth finite-volume/time weak balances at measured axial stations.

No interpolation between stations. Meridional flux divergence is NOT included;
callers must qualify its omission separately. Boundary-cell weights are zero,
so these rows do not estimate boundary transfer. Full depth sampling required.
"""
import numpy as np
from scipy.integrate import simpson


def project_sensor_depth_weak(*,t,x,u,diffusivity_features,reaction_features,volume,area_x,distance_x,window=31,stride=15):
    time,depth=np.asarray(t,dtype=float),np.asarray(x,dtype=float)
    values=np.asarray(u,dtype=float)
    diffusion,reactions=map(lambda v:np.asarray(v,dtype=float),(diffusivity_features,reaction_features))
    if (time.ndim!=1 or depth.ndim!=1 or len(depth)<3 or len(time)<window
            or not np.all(np.isfinite(time)) or not np.all(np.isfinite(depth))
            or np.any(np.diff(time)<=0) or np.any(np.diff(depth)<=0)):
        raise ValueError("increasing finite time and depth coordinates required")
    if values.ndim!=3 or values.shape[0]!=len(time) or values.shape[2]!=len(depth):
        raise ValueError("time-by-station-by-depth field required")
    if (diffusion.ndim!=4 or reactions.ndim!=4 or diffusion.shape[:3]!=values.shape
            or reactions.shape[:3]!=values.shape or diffusion.shape[-1]<1):
        raise ValueError("aligned finite diffusivity/reaction feature arrays required")
    volumes,areas,distances=map(lambda v:np.asarray(v,dtype=float),(volume,area_x,distance_x))
    if volumes.shape!=values.shape[1:] or areas.shape!=(values.shape[1],values.shape[2]-1) or distances.shape!=(values.shape[2]-1,):
        raise ValueError("declared station cell geometry must match observed depth grid")
    if (any(not np.all(np.isfinite(v)) for v in (values,diffusion,reactions,volumes,areas,distances))
            or np.any(volumes<=0) or np.any(areas<=0) or np.any(distances<=0)
            or not np.allclose(distances,np.diff(depth),rtol=1e-10,atol=1e-14)):
        raise ValueError("finite fields and positive consistent cell geometry required")
    if not isinstance(window,int) or not isinstance(stride,int) or window<3 or stride<1:
        raise ValueError("valid integer window/stride required")
    q=(depth-(depth[0]+depth[-1])/2)/((depth[-1]-depth[0])/2)
    spatial=(1-q*q)**4
    weights=volumes*spatial[None,:]
    face_diffusion=.5*(diffusion[:,:,1:,:]+diffusion[:,:,:-1,:])
    linear_flux=-face_diffusion*np.diff(values,axis=2)[...,None]/distances[None,None,:,None]
    pair=np.sum(linear_flux*areas[None,:,:,None]*np.diff(spatial)[None,None,:,None],axis=2)
    sources=np.sum(reactions*weights[None,:,:,None],axis=2)
    weighted=np.sum(values*weights[None,:,:],axis=2)
    rows,targets=[],[]
    for start in range(0,len(time)-window+1,stride):
        tt=time[start:start+window];half=(tt[-1]-tt[0])/2
        qt=(tt-(tt[0]+tt[-1])/2)/half;base=1-qt*qt
        phi,derivative=base**4,-8*qt*base**3/half
        target=-simpson((weighted[start:start+window]-weighted[start])*derivative[:,None],x=tt,axis=0)
        drows=simpson(pair[start:start+window]*phi[:,None,None],x=tt,axis=0)
        rrows=simpson(sources[start:start+window]*phi[:,None,None],x=tt,axis=0)
        rows.append(np.column_stack((drows,rrows)));targets.append(target)
    return np.vstack(rows),np.concatenate(targets)
