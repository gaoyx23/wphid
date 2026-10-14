"""Constrained variable projection and a source-locked conditional Adam adapter.

Both programs use exactly the same frozen metric, physical parameter box and
training-only coefficient scales. Adam equations/hyperparameters follow the
recorded official source; graph enumeration and weak inputs are adaptations.
There is no sparsity penalty inside supplied support and no post-Adam profile.
"""

import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear, minimize
from scipy.sparse import block_diag

from wphid.source_geometry_v6 import profile_information
from wphid.coupled_solver_v1.model import parameter_box


def common_metric(problems, reference_sigma):
    """Fixed reference measurement metric, NOT the actual residual covariance.

Nominal channel resolutions define units including deterministic coordinates.
No empirical/truth variance, fitted weights, jitter or V5 anchor scan is used.
The full window correlation of the reference storage map is retained. Sources,
transports and data-dependent design noise are excluded from this approximation
and must be reported; actual observation covariance has a separate diagnostic.
"""
    matrices = []
    for p in problems:
        source = p.projector.tests.derivative
        spatial = p.projector.spatial.reshape(p.projector.row_shape[1],-1)*p.projector.mesh.volume.ravel()
        unit = np.kron(source @ source.T, spatial @ spatial.T)
        for field in ("carbon","pO2","pH2O","pProducts","T"):
            matrices.append(np.linalg.cholesky(unit*reference_sigma[field]**2))
    lower = block_diag(matrices, format="csr")
    # Each observation packet is independent here; no shared error is dropped
    # from an actual covariance: this matrix is explicitly a fixed reference.
    return Metric(matrices, lower.shape[0])


class Metric:
    def __init__(self, lower_blocks, rows):
        self.blocks, self.rows = lower_blocks, rows
    def apply(self, values):
        result=[]; start=0
        for L in self.blocks:
            result.append(solve_triangular(L, values[start:start+len(L)], lower=True, check_finite=False))
            start+=len(L)
        if start != len(values):
            raise ValueError("common metric row identity mismatch")
        return np.concatenate(result,axis=0)


def stacked(problems, candidate, point):
    models=[p.assemble(candidate,point) for p in problems]
    keys=models[0].column_keys
    if any(m.column_keys!=keys for m in models):
        raise ValueError("candidate column identities vary across conditions")
    return (np.concatenate([m.target for m in models]),np.vstack([m.matrix for m in models]),
        {k:np.vstack([m.matrix_jacobian.get(k,np.zeros_like(m.matrix)) for m in models]) for k in point},keys)


def setup(problems,candidate,metric):
    box=parameter_box(candidate); names=tuple(box)
    low=np.array([box[k][0] for k in names]); span=np.array([box[k][1]-box[k][0] for k in names])
    point=dict(zip(names,low+.5*span))
    b,G,_,keys=stacked(problems,candidate,point)
    b=metric.apply(b); G=metric.apply(G)
    norms=np.linalg.norm(G,axis=0)
    # Zero columns remain zero. This is a numerical coordinate scale, not a
    # ridge and not evidence that any parameter is identifiable.
    scales=np.where(norms>0,1/norms,1.)
    upper=np.full(len(keys),np.inf)
    if "heat_capacity_contrast" in keys:
        upper[keys.index("heat_capacity_contrast")]=2./scales[keys.index("heat_capacity_contrast")]
    return names,low,span,keys,scales,upper,b


def evaluate(problems,candidate,metric,names,low,span,scales,x):
    point=decode_unit_box(candidate,x)
    b,G,dG,keys=stacked(problems,candidate,point)
    return metric.apply(b),metric.apply(G)*scales,{k:metric.apply(v)*scales for k,v in dG.items()},keys


def fit_profile(problems,candidate,metric,*,cpu_cap=10.,max_calls=2000,starts=(.25,.5,.75),max_iterations=400):
    began=time.perf_counter(); cpu=time.process_time(); calls=0; best=None; records=[]
    names,low,span,keys,scales,upper,_=setup(problems,candidate,metric)
    class BudgetStop(Exception): pass
    def objective(x):
        nonlocal calls,best
        if calls>=max_calls or time.process_time()-cpu>=cpu_cap: raise BudgetStop
        calls+=1
        b,G,dG,_=evaluate(problems,candidate,metric,names,low,span,scales,x)
        fit=lsq_linear(G,b,bounds=(0.,upper),tol=1e-10,max_iter=100,lsq_solver="exact")
        if not fit.success: raise ValueError("inner constrained profile failed")
        residual=b-G@fit.x; value=float(residual@residual)
        grad=np.array([-2*(dG.get(k,np.zeros_like(G))@fit.x)@residual*s for k,s in zip(names,span)])
        if best is None or value<best[0]: best=(value,x.copy(),fit.x.copy())
        return value,grad
    errors=[]
    for fraction in starts:
        x=np.full(len(names),fraction)
        try:
            result=minimize(objective,x,jac=True,method="L-BFGS-B",bounds=[(0.,1.)]*len(x),options={"maxiter":max_iterations,"ftol":1e-12,"gtol":1e-8,"maxls":30})
            records.append({"start_fraction":fraction,"success":bool(result.success),"message":str(result.message),"iterations":int(result.nit)})
        except BudgetStop:
            records.append({"start_fraction":fraction,"status":"budget_exhausted"});break
        except (ValueError,FloatingPointError) as error:
            errors.append(str(error));records.append({"start_fraction":fraction,"status":"domain_or_solver_failure","error":str(error)})
    return finish(best,names,low,span,keys,scales,calls,cpu,began,records,errors,"WPHID_CONSTRAINED_PROFILE_FIXED_METRIC_V2_NUMERIC_BOUND",candidate)


def fit_adam(problems,candidate,metric,*,cpu_cap=10.,max_calls=40000,starts=(.25,.5,.75),learning_rate=.01):
    began=time.perf_counter();cpu=time.process_time();calls=0;best=None;records=[];errors=[]
    names,low,span,keys,scales,upper,_=setup(problems,candidate,metric)
    for fraction in starts:
        x=np.full(len(names),fraction)
        try:
            b,G,_,_=evaluate(problems,candidate,metric,names,low,span,scales,x)
            initial=lsq_linear(G,b,bounds=(0.,upper),tol=1e-10,max_iter=100)
            if not initial.success: raise ValueError("common training-only coefficient initialization failed")
            c=initial.x.copy();v=np.r_[x,c];m=np.zeros_like(v);s=np.zeros_like(v);epochs=0
            for epoch in range(max_calls//len(starts)):
                if calls>=max_calls or time.process_time()-cpu>=cpu_cap:break
                calls+=1;epochs+=1
                b,G,dG,_=evaluate(problems,candidate,metric,names,low,span,scales,v[:len(names)])
                residual=b-G@v[len(names):]
                gradient=np.r_[[-2*(dG.get(k,np.zeros_like(G))@v[len(names):])@residual*unit for k,unit in zip(names,span)],-2*G.T@residual]
                # Rescale by row count, as in a mean-squared-error objective.
                gradient/=len(b)
                m=.9*m+.1*gradient;s=.99*s+.01*gradient**2
                lr=learning_rate*.5**(epoch//3000)
                v-=lr*(m/(1-.9**epochs))/(np.sqrt(s/(1-.99**epochs))+1e-15)
                v[:len(names)]=np.clip(v[:len(names)],0.,1.)
                v[len(names):]=np.minimum(np.maximum(v[len(names):],0.),upper)
            b,G,_,_=evaluate(problems,candidate,metric,names,low,span,scales,v[:len(names)])
            residual=b-G@v[len(names):];value=float(residual@residual)
            # Each restart is selected by its own terminal score. No WPHID
            # refinement or best-epoch selection is used.
            if best is None or value<best[0]:best=(value,v[:len(names)].copy(),v[len(names):].copy())
            records.append({"start_fraction":fraction,"epochs":epochs,"budget_exhausted":time.process_time()-cpu>=cpu_cap or calls>=max_calls})
            if time.process_time()-cpu>=cpu_cap or calls>=max_calls:break
        except (ValueError,FloatingPointError) as error:
            errors.append(str(error));records.append({"start_fraction":fraction,"status":"domain_or_solver_failure","error":str(error)})
    return finish(best,names,low,span,keys,scales,calls,cpu,began,records,errors,"ADAM_SINDY_CONDITIONAL_FIXED_METRIC_V2_NUMERIC_BOUND",candidate)


def finish(best,names,low,span,keys,scales,calls,cpu,began,records,errors,version,candidate):
    result={"version":version,"calls":calls,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-began,
        "restarts":records,"errors":errors,"fitted":best is not None,"exact_MLE":False,"native_end_to_end_ADAM":False}
    if best is not None:
        value,x,c=best
        result.update({"objective":value,"parameters":decode_unit_box(candidate,x),
            "coefficients":dict(zip(keys,(c*scales).tolist())),"parameter_count":len(names)+len(keys),
            "post_fit_profile_refinement":False})
    return result


def information_at_fit(problems,candidate,metric,fit):
    point=fit["parameters"];_,G,dG,keys=stacked(problems,candidate,point)
    a=np.array([fit["coefficients"][k] for k in keys]);names=tuple(point)
    derivatives=metric.apply(np.column_stack([dG[k]@a for k in names]))
    kinetic=[i for i,k in enumerate(names) if k.startswith("kinetic.")]
    nuisance=[i for i in range(len(names)) if i not in kinetic]
    if not kinetic:return {"kinetic_count":0,"kinetic_rank":0,"nuisance_only":True}
    units=np.array([parameter_box(candidate)[names[i]][1]-parameter_box(candidate)[names[i]][0] for i in kinetic])
    info=profile_information(derivatives[:,kinetic],np.column_stack((metric.apply(G),derivatives[:,nuisance])),parameter_scale=units,rtol=1e-8)
    return {"kinetic_count":len(kinetic),"kinetic_rank":info.rank,"nuisance_rank":info.nuisance_rank,
        "singular_values":info.singular_values.tolist(),"condition_ratio":info.condition_ratio,
        "all_allowed_nuisance_coordinates_included":True,"confidence_interval":False}



def decode_unit_box(candidate, unit):
    """Exact inclusive endpoints, with no tolerance expansion or data clipping."""
    box=parameter_box(candidate);names=tuple(box);unit=np.asarray(unit,dtype=float)
    if unit.shape!=(len(names),) or not np.all(np.isfinite(unit)) or np.any(unit<0) or np.any(unit>1):
        raise ValueError("unit coordinates must lie in the declared closed box")
    result={}
    for key,value in zip(names,unit):
        low,high=box[key]
        physical=low+(high-low)*float(value)
        # Conversion from an already-valid unit coordinate can round one ulp
        # above the declared upper endpoint. Bound the representation itself;
        # values genuinely outside the unit domain remain rejected.
        result[key]=min(max(physical,low),high)
    return result
