"""Artificial software checks only; no saved engineering input or batch."""

import numpy as np
import pytest

from wphid.coupled_balance_v1 import ShellMesh, WeakProjector, temporal_tests
from wphid.coupled_balance_v1.assembler import BoundaryInputs, FIELDS, PRESSURES
from wphid.coupled_balance_v1.operators import SIDES, Modulation
from wphid.coupled_solver_v1 import Candidate, ValueProblem, candidate_space, parameter_box, common_metric
from wphid.coupled_solver_v1.solver import setup, evaluate, fit_profile, fit_adam
from wphid.coupled_solver_v1.uncertainty import raw_at_fit


def problem():
    nt,ns,nx=9,2,2;t,s,x=np.indices((nt,ns,nx))
    mesh=ShellMesh(np.full((ns,nx),.7),np.full((ns-1,nx),.8),np.full((ns,nx-1),.9),
        np.full((ns-1,nx),.4),np.full((ns,nx-1),.3),{k:np.full(ns if k in ("inner","outer") else nx,.5) for k in SIDES})
    projector=WeakProjector(mesh,temporal_tests(np.arange(nt,dtype=float),window_samples=5,stride_samples=2),np.eye(ns*nx).reshape(ns*nx,ns,nx))
    fields={"T":650.+17*t+22*s+31*x,"carbon":.9-.002*t-.003*s-.004*x,
        "pO2":2e4+100*t+200*s+300*x,"pH2O":6e5+200*t+100*s+300*x,
        "pProducts":1e3+40*t+20*s+10*x,"crack":.1+.003*t+.01*s+.02*x,"stress":2e7+2e5*t+3e5*s}
    loads={"internal_oxygen_pressure":np.full((nt,ns),28000.),"external_steam_pressure":np.full((nt,ns),7e5),
        "external_oxygen_pressure":np.full((nt,ns),21000.),"surface_heat_flux":np.full((nt,ns),100.)}
    ext={("pO2","inner"):loads["internal_oxygen_pressure"],("pH2O","inner"):loads["external_steam_pressure"],
        ("pProducts","inner"):np.zeros((nt,ns)),("pO2","outer"):loads["external_oxygen_pressure"]}
    boundary=BoundaryInputs(ext,{k:Modulation(np.ones((nt,ns))) for k in ext},frozenset((f,k) for f in PRESSURES for k in SIDES if (f,k) not in ext),
        {"inner":loads["surface_heat_flux"]},{"outer":np.full((nt,ns),300.)},frozenset(("inlet","outlet")))
    p=ValueProblem(projector,fields,boundary);p.raw_loads=loads
    return p


@pytest.mark.parametrize("candidate",candidate_space(),ids=lambda c:c.identity)
def test_new_candidates_match_qualified_raw_values_or_independent_column_count(candidate):
    p=problem();point={k:.5*(v[0]+v[1]) for k,v in parameter_box(candidate).items()}
    sigma={k:np.ones_like(v) for k,v in p.fields.items()};ls={k:np.ones_like(v) for k,v in p.raw_loads.items()}
    model,lin,factor,_=raw_at_fit(p,candidate,point,sigma,ls)
    assert len(lin.column_jacobians)==len(model.column_keys)
    assert np.all(np.isfinite(model.matrix))
    assert len(model.column_keys)==(14 if candidate.source=="both_independent" else len(model.column_keys))
    assert factor.shape[0]==len(lin.coordinate_keys)


def test_all_outer_derivatives_match_independent_difference():
    p=problem();candidate=Candidate("both_shared");point={k:.5*(v[0]+v[1]) for k,v in parameter_box(candidate).items()}
    model=p.assemble(candidate,point)
    for k,(low,high) in parameter_box(candidate).items():
        step=1e-5*(high-low);plus=dict(point);minus=dict(point);plus[k]+=step;minus[k]-=step
        reference=(p.assemble(candidate,plus).matrix-p.assemble(candidate,minus).matrix)/(2*step)
        scale=max(np.max(np.abs(reference)),1e-20)
        np.testing.assert_allclose(model.matrix_jacobian.get(k,np.zeros_like(reference))/scale,reference/scale,rtol=1e-5,atol=2e-7)


def test_metric_keeps_overlap_and_units_without_noise_variance_claim():
    p=problem();sig={k:1. for k in FIELDS};metric=common_metric([p],sig)
    L=metric.blocks[0]
    assert np.count_nonzero(L-np.diag(np.diag(L)))>0
    values=np.arange(metric.rows,dtype=float)
    out=metric.apply(values)
    assert np.all(np.isfinite(out))
    with pytest.raises(ValueError):metric.apply(values[:-1])


def test_profile_envelope_gradient_includes_constitutive_nuisance():
    from scipy.optimize import lsq_linear
    p=problem();c=Candidate("both_shared");metric=common_metric([p],{k:1. for k in FIELDS})
    names,low,span,keys,scales,upper,b=setup([p],c,metric);x=np.full(len(names),.4)
    b,G,dG,_=evaluate([p],c,metric,names,low,span,scales,x)
    coeff=lsq_linear(G,b,bounds=(0.,upper),tol=1e-12,max_iter=200).x
    residual=b-G@coeff
    for i,k in enumerate(names):
        h=1e-5;plus=x.copy();minus=x.copy();plus[i]+=h;minus[i]-=h
        values=[]
        for v in (plus,minus):
            bb,GG,_,_=evaluate([p],c,metric,names,low,span,scales,v)
            a=lsq_linear(GG,bb,bounds=(0.,upper),tol=1e-12,max_iter=200).x
            values.append(np.sum((bb-GG@a)**2))
        numeric=(values[0]-values[1])/(2*h)
        actual=-2*(dG.get(k,np.zeros_like(G))@coeff)@residual*span[i]
        np.testing.assert_allclose(actual,numeric,rtol=3e-3,atol=2e-5*max(abs(numeric),1.))


def test_both_programs_use_bounds_and_keep_failures_and_budget():
    p=problem();c=Candidate("null");metric=common_metric([p],{k:1. for k in FIELDS})
    for program in (fit_profile,fit_adam):
        fit=program([p],c,metric,cpu_cap=3.,max_calls=5,starts=(.25,))
        assert fit["calls"]<=5
        assert "restarts" in fit and "errors" in fit
        if fit["fitted"]:
            for k,value in fit["parameters"].items():
                assert parameter_box(c)[k][0]<=value<=parameter_box(c)[k][1]
            assert min(fit["coefficients"].values())>=0
            assert not fit["post_fit_profile_refinement"]
