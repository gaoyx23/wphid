"""Conditional reaction-prefactor profiling in discrete gas weak balances.

This is an errors-in-variables development diagnostic, NOT calibrated support
selection. Temperature, carbon and gas fields must be observed/reconstructed;
activation energies, pressure exponents and transport coefficients stay fixed.
"""
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import nnls

from .axisymmetric import axisymmetric_window_slices
from .bell_shell import _bump
from .empirical_noise import whiten_noise

SUPPORTS = {"no_reaction":(), "oxygen_only":(0,), "steam_only":(1,), "oxygen_and_steam":(0,1)}


def project_reaction_source(*, forcing, t, s, x, volume, window_shape, stride_shape):
    """Integral of a source against exactly the discrete transport test weights.

    If forcing is one reaction's signed gas source, this equals
    r(omit reaction)-r(include reaction). No pressure derivative is needed.
    """
    coords = tuple(np.asarray(v,dtype=float) for v in (t,s,x))
    if any(v.ndim!=1 or len(v)<3 or not np.all(np.isfinite(v)) or np.any(np.diff(v)<=0) for v in coords):
        raise ValueError("finite increasing coordinates required")
    shape=tuple(len(v) for v in coords)
    source=np.asarray(forcing,dtype=float);vol=np.asarray(volume,dtype=float)
    if source.shape!=shape or not np.all(np.isfinite(source)) or vol.shape!=shape[1:] or not np.all(np.isfinite(vol)) or np.any(vol<=0):
        raise ValueError("aligned finite forcing and positive volume required")
    result=[]
    for slices in axisymmetric_window_slices(shape,window_shape,stride_shape):
        tt,ss,xx=[v[sl] for v,sl in zip(coords,slices)]
        bt,_,_=_bump(tt);bs,_,_=_bump(ss);bx,_,_=_bump(xx)
        spatial=np.zeros(shape[1:]);spatial[slices[1],slices[2]]=bs[:,None]*bx[None,:]
        result.append(simpson(np.sum(source[slices[0]]*spatial*vol,axis=(1,2))*bt,x=tt))
    if not result:raise ValueError("window configuration produced no source balances")
    return np.asarray(result)


def fit_reaction_profile(*, full_reaction_residual, omission_columns, metric, support):
    """Fit nonnegative multipliers of reference oxygen/steam reaction rates.

    For observed fields, r(a) = r(full) + Delta_O + Delta_W - Delta @ a.
    Zero-centered whitening is only a FIXED weighting convention. It is not a
    profile likelihood or a noise law after fitting state-dependent columns.
    """
    residual=np.asarray(full_reaction_residual,dtype=float)
    columns=np.asarray(omission_columns,dtype=float)
    if support not in SUPPORTS or residual.ndim!=1 or columns.shape!=(len(residual),2):
        raise ValueError("declared support and aligned two-reaction columns required")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(columns)):
        raise ValueError("finite weak data required")
    zero_metric=dict(metric,center=np.zeros(len(residual)))
    target=residual+columns.sum(axis=1)
    yw=whiten_noise(target[None,:],zero_metric)[0]
    xw=whiten_noise(columns.T,zero_metric).T
    active=SUPPORTS[support];amplitudes=np.zeros(2)
    rank=0;condition=None;singular=[]
    if active:
        matrix=xw[:,active];scale=np.linalg.norm(matrix,axis=0)
        usable=scale>0
        normalized=np.zeros_like(matrix)
        normalized[:,usable]=matrix[:,usable]/scale[usable]
        singular=np.linalg.svd(normalized,compute_uv=False)
        rank=int(np.linalg.matrix_rank(normalized))
        condition=float(singular[0]/singular[-1]) if rank==len(active) else None
        if np.any(usable):
            beta,_=nnls(normalized[:,usable],yw)
            indices=np.asarray(active)[usable]
            amplitudes[indices]=beta/scale[usable]
    profiled=target-columns@amplitudes
    white=whiten_noise(profiled[None,:],zero_metric)[0]
    return dict(support=support,amplitude_multipliers=amplitudes,
        weighted_squared_residual=float(white@white),profiled_residual=profiled,
        normalized_design_rank=rank,normalized_design_condition=condition,
        normalized_design_singular_values=singular,
        fitted_active_coefficients_at_zero=[int(i) for i in active if amplitudes[i]==0],
        parameter_estimates_are_development_diagnostics=True,
        calibrated_support_selection=False,parameter_confidence_intervals=None,
        unknown_parameters_scope="oxygen/steam prefactor multipliers only; fixed rate shapes and transport coefficients")
