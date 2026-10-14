"""Gaussian covariance envelope + explicit deterministic surrogate error tube.

For pressures in the anchor triangle plus a normalized L2 discrepancy <= B:
point-noise variance is bounded by sigma_upper**2. For each of two common gain
modes, (u+e)(u+e)' <= 2uu' + 2ee', and ||e|| <= .01 B.
Jensen bounds the interpolated u by the sum of its three anchor outer products.
Thus C <= diag(sigma_upper) [c I + 2 sum(u_anchor u_anchor')] diag(sigma_upper),
c=1+4(.01B)^2. A fitted forward residual is bounded by noise plus tube error;
the acceptance radius is sqrt(chi2(n,.95)) + B/sqrt(c).
The bound is conditional on the tube holding; it does not certify the PDE
surrogate throughout a continuous parameter domain.
"""
import numpy as np
from scipy.stats import chi2
from .profiled_forward import build_forward_template


def make_envelope_template(*,reference,oxygen_anchor,steam_anchor,relative_noise=.01,
        surrogate_rms_budget=.25,nominal_coverage=.95):
    arrays=[np.asarray(v,dtype=float).reshape(-1) for v in (reference,oxygen_anchor,steam_anchor)]
    n=len(arrays[0])
    if n%3 or not n or any(v.shape!=(n,) or np.any(v<=0) or not np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("positive aligned three-gas reference/anchor pressures required")
    if not 0<relative_noise<1 or not 0<surrogate_rms_budget or not 0<nominal_coverage<1:
        raise ValueError("positive explicit tube/noise and valid coverage required")
    reference_scale=relative_noise*arrays[0]
    budget=surrogate_rms_budget*np.sqrt(n)
    maximum=np.max(arrays,axis=0)
    upper=relative_noise*(maximum+budget*reference_scale)
    diagonal_inflation=1+4*(relative_noise*budget)**2
    oxygen_mask=np.r_[np.ones(n//3),np.zeros(2*n//3)]
    modes=np.column_stack([relative_noise*a/upper*mask for a in arrays for mask in (np.ones(n),oxygen_mask)])
    effective_scale=upper*np.sqrt(diagonal_inflation)
    effective_modes=modes*np.sqrt(2/diagonal_inflation)
    template=build_forward_template(reference=arrays[0],oxygen_anchor=arrays[1],steam_anchor=arrays[2],
        scale=effective_scale,modes=effective_modes)
    radius=np.sqrt(chi2.ppf(nominal_coverage,n))+budget/np.sqrt(diagonal_inflation)
    template.update(reference_pressure_noise_scale=reference_scale,surrogate_l2_budget=float(budget),
        surrogate_rms_budget=float(surrogate_rms_budget),covariance_diagonal_inflation=float(diagonal_inflation),
        squared_acceptance_radius=float(radius**2),nominal_coverage=float(nominal_coverage),
        acceptance_rule="Gaussian upper covariance and deterministic surrogate-error triangle inequality",
        domain_validity="conditional on local physical surrogate discrepancy lying inside declared L2 tube")
    return template


def qualify_surrogate(actual_pressure,predicted_pressure,template):
    error=(np.asarray(actual_pressure)-np.asarray(predicted_pressure))/np.asarray(template["reference_pressure_noise_scale"])
    rms=float(np.sqrt(np.mean(error**2)))
    return dict(normalized_bias_rms=rms,declared_rms_budget=template["surrogate_rms_budget"],
        target_met=rms<=template["surrogate_rms_budget"],continuous_domain_certified=False)
