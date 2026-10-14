"""Exact local pressure Jacobian of known-driver discrete weak balances.

Analytic derivative of the nonlinear face flux, not a nonlinear noise law.
Reaction-source pressure derivatives must be declared by the caller.
"""
import numpy as np
from scipy.integrate import simpson

from ..transport_mechanisms import modulation_fraction
from .discrete_transport import project_discrete_transport_candidate
from .axisymmetric import axisymmetric_window_slices
from .bell_shell import _bump

PRESSURES=("pO2","pH2O","pProducts")


def discrete_pressure_jacobian(*,forcing_pressure_derivatives,**kwargs):
    result=project_discrete_transport_candidate(**kwargs)
    fields=kwargs["observed_fields"];species=kwargs["species"]
    pressure=np.asarray(fields[species],dtype=float);shape=pressure.shape
    time,s,x=[np.asarray(kwargs[n],dtype=float) for n in ("t","s","x")]
    geometry=kwargs["geometry"]
    if set(forcing_pressure_derivatives)!=set(PRESSURES):
        raise ValueError("explicit source derivative for every pressure required")
    derivatives={n:np.asarray(v,dtype=float) for n,v in forcing_pressure_derivatives.items()}
    if any(v.shape!=shape or not np.all(np.isfinite(v)) for v in derivatives.values()):
        raise ValueError("finite aligned source pressure derivatives required")
    candidate=kwargs["candidate"];transport=candidate.configure_transport(kwargs["transport"])
    fraction=modulation_fraction(candidate.modulation,temperature=fields["T"],stress=fields["stress"],
        carbon=fields["carbon"],crack=fields["crack"],transport=transport,mechanics=kwargs["mechanics"])
    d0=getattr(transport,{"pO2":"oxygen_diffusivity_closed","pH2O":"steam_diffusivity_closed",
        "pProducts":"product_diffusivity_closed"}[species])
    d=d0*(1+(transport.open_to_closed_diffusivity_ratio-1)*fraction)
    metric=np.asarray(geometry.get("meridional_face_metric",np.ones((shape[1]-1,shape[2]))))
    ks=(.5*(d[:,1:,:]+d[:,:-1,:])*transport.meridional_to_through_diffusivity_ratio/
        (np.asarray(geometry["distance_s"])[None,:,None]*metric[None,:,:]))
    kx=.5*(d[:,:,1:]+d[:,:,:-1])/np.asarray(geometry["distance_x"])[None,None,:]
    qs,qx=-ks*np.diff(pressure,axis=1),-kx*np.diff(pressure,axis=2)
    alpha=transport.forchheimer_inverse_flux_scale
    flux_derivative_s=1/np.sqrt(1+4*alpha*np.abs(qs))
    flux_derivative_x=1/np.sqrt(1+4*alpha*np.abs(qx))
    jacobians={n:np.zeros((len(result["residual"]),)+shape) for n in PRESSURES}
    for row,slices in enumerate(axisymmetric_window_slices(shape,kwargs["window_shape"],kwargs["stride_shape"])):
        tt,ss,xx=[v[sl] for v,sl in zip((time,s,x),slices)]
        bt,bt_t,_=_bump(tt);bs,_,_=_bump(ss);bx,_,_=_bump(xx)
        spatial=np.zeros(shape[1:]);spatial[slices[1],slices[2]]=bs[:,None]*bx[None,:]
        quadrature=simpson(np.eye(len(tt)),x=tt,axis=1)
        w=(quadrature*bt)[:,None,None]
        volume_test=spatial*np.asarray(geometry["volume"])
        for n in PRESSURES:
            jacobians[n][row,slices[0]]-=w*volume_test*derivatives[n][slices[0]]
        primary=jacobians[species][row]
        primary[slices[0]]-=(quadrature*bt_t)[:,None,None]*volume_test
        bs_flux=w*ks[slices[0]]*flux_derivative_s[slices[0]]*np.asarray(geometry["area_s"])*np.diff(spatial,axis=0)
        bx_flux=w*kx[slices[0]]*flux_derivative_x[slices[0]]*np.asarray(geometry["area_x"])*np.diff(spatial,axis=1)
        primary[slices[0],:-1,:]-=bs_flux;primary[slices[0],1:,:]+=bs_flux
        primary[slices[0],:,:-1]-=bx_flux;primary[slices[0],:,1:]+=bx_flux
    return dict(residual=result["residual"],pressure_jacobians=jacobians,
        candidate_id=result["candidate_id"],first_order_only=True,
        nonpressure_drivers_held_fixed=True,calibrated_selection=False)


def propagate_independent_pressure_noise(*,pressure_jacobians,standard_deviations):
    if set(pressure_jacobians)!=set(PRESSURES) or set(standard_deviations)!=set(PRESSURES):
        raise ValueError("all pressure channels and noise scales must be declared")
    covariance=None
    row_count=None
    for name in PRESSURES:
        jacobian=np.asarray(pressure_jacobians[name],dtype=float)
        std=np.asarray(standard_deviations[name],dtype=float)
        if jacobian.ndim!=4 or std.shape!=jacobian.shape[1:] or not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(std)) or np.any(std<0):
            raise ValueError("finite aligned Jacobians and nonnegative noise scales required")
        if row_count is not None and row_count!=len(jacobian):
            raise ValueError("Jacobian residual rows must align across pressures")
        row_count=len(jacobian)
        weighted=(jacobian*std[None,:,:,:]).reshape(row_count,-1)
        contribution=weighted@weighted.T
        covariance=contribution if covariance is None else covariance+contribution
    return covariance
