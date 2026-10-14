"""Linear Darcy dictionary in explicit two-dimensional finite-volume balances.

Geometry and aligned measured/reconstructed fields are caller supplied. Both
spatial directions retained; boundary exchange not identified by vanishing tests.
No constitutive gains or source coefficients are read from hidden trajectories.
"""
import numpy as np
from scipy.integrate import simpson
from .axisymmetric import axisymmetric_window_slices
from .bell_shell import _bump


def project_linear_discrete_transport(*,t,s,x,u,diffusivity_features,reaction_features,
        geometry,window_shape,stride_shape):
    coordinates=[np.asarray(v,dtype=float) for v in (t,s,x)]
    if any(v.ndim!=1 or len(v)<3 or not np.all(np.isfinite(v)) or np.any(np.diff(v)<=0) for v in coordinates):
        raise ValueError("finite increasing coordinates required")
    shape=tuple(len(v) for v in coordinates)
    values=np.asarray(u,dtype=float)
    diffusion=np.asarray(diffusivity_features,dtype=float);reactions=np.asarray(reaction_features,dtype=float)
    if (values.shape!=shape or diffusion.ndim!=4 or diffusion.shape[:3]!=shape or diffusion.shape[-1]<1
        or reactions.ndim!=4 or reactions.shape[:3]!=shape
        or any(not np.all(np.isfinite(v)) for v in (values,diffusion,reactions))):
        raise ValueError("finite aligned state and features required")
    expected=dict(volume=shape[1:],area_s=(shape[1]-1,shape[2]),area_x=(shape[1],shape[2]-1),
        distance_s=(shape[1]-1,),distance_x=(shape[2]-1,))
    mesh={}
    for name,dims in expected.items():
        value=np.asarray(geometry.get(name),dtype=float)
        if value.shape!=dims or not np.all(np.isfinite(value)) or np.any(value<=0):
            raise ValueError("positive finite geometry required: "+name)
        mesh[name]=value
    for name,coordinate in (("distance_s",coordinates[1]),("distance_x",coordinates[2])):
        if not np.allclose(mesh[name],np.diff(coordinate),rtol=1e-10,atol=1e-14):raise ValueError("inconsistent cell distances")
    metric=np.asarray(geometry.get("meridional_face_metric",np.ones(expected["area_s"])),dtype=float)
    if metric.shape!=expected["area_s"] or not np.all(np.isfinite(metric)) or np.any(metric<=0):
        raise ValueError("positive finite declared meridional metric required")
    js=-.5*(diffusion[:,1:]+diffusion[:,:-1])*np.diff(values,axis=1)[...,None]/(mesh["distance_s"][:,None]*metric)[None,...,None]
    jx=-.5*(diffusion[:,:,1:]+diffusion[:,:,:-1])*np.diff(values,axis=2)[...,None]/mesh["distance_x"][None,None,:,None]
    rows=[];targets=[]
    for slices in axisymmetric_window_slices(shape,window_shape,stride_shape):
        tt,ss,xx=[v[sl] for v,sl in zip(coordinates,slices)]
        bt,bt_t,_=_bump(tt);bs,_,_=_bump(ss);bx,_,_=_bump(xx)
        test=np.zeros(shape[1:]);test[slices[1],slices[2]]=bs[:,None]*bx[None,:]
        weight=test*mesh["volume"]
        target=-simpson(np.sum(values[slices[0]]*weight,axis=(1,2))*bt_t,x=tt)
        flux=np.sum(js[slices[0]]*mesh["area_s"][None,...,None]*np.diff(test,axis=0)[None,...,None],axis=(1,2))
        flux+=np.sum(jx[slices[0]]*mesh["area_x"][None,...,None]*np.diff(test,axis=1)[None,...,None],axis=(1,2))
        source=np.sum(reactions[slices[0]]*weight[None,...,None],axis=(1,2))
        rows.append(np.concatenate((simpson(flux*bt[:,None],x=tt,axis=0),simpson(source*bt[:,None],x=tt,axis=0))))
        targets.append(target)
    if not rows:raise ValueError("empty weak configuration")
    return np.asarray(rows),np.asarray(targets)
