import numpy as np
from wphid.discovery.forward_envelope import make_envelope_template,qualify_surrogate
from wphid.discovery.profiled_forward import score_forward


def fixture():
    ref=np.linspace(1,2,12);oxy=ref*np.linspace(.85,.98,12);steam=ref*np.linspace(1.01,1.2,12)
    return ref,oxy,steam,make_envelope_template(reference=ref,oxygen_anchor=oxy,steam_anchor=steam)


def test_covariance_envelope_psd_dominates_noisy_interior_with_tube_discrepancy():
    ref,oxy,steam,template=fixture();rng=np.random.default_rng(323)
    sigma=np.asarray(template["scale"]);modes=np.asarray(template["modes"])
    envelope=sigma[:,None]*(np.eye(12)+modes@modes.T)*sigma[None,:]
    mask=np.r_[np.ones(4),np.zeros(8)]
    for _ in range(40):
        weights=rng.dirichlet(np.ones(3));mean=weights[0]*ref+weights[1]*oxy+weights[2]*steam
        direction=rng.normal(size=12);direction*=template["surrogate_l2_budget"]/np.linalg.norm(direction)
        actual=mean+.01*ref*direction
        gain=.01*actual;oxygen_gain=gain*mask
        covariance=np.diag(gain**2)+np.outer(gain,gain)+np.outer(oxygen_gain,oxygen_gain)
        assert np.linalg.eigvalsh(envelope-covariance).min()>-1.e-12


def test_anchor_and_interior_clean_forward_residual_pass_fixed_envelope():
    ref,oxy,steam,template=fixture()
    for mean in (ref,oxy,steam,.25*ref+.5*oxy+.25*steam):
        assert score_forward(mean,template)["weighted_squared_residual"]<template["squared_acceptance_radius"]
    assert template["squared_acceptance_radius"]>0


def test_surrogate_qualification_cannot_certify_entire_domain():
    ref,_,_,template=fixture()
    result=qualify_surrogate(ref+.001*ref,ref,template)
    assert result["target_met"] and not result["continuous_domain_certified"]
    assert not qualify_surrogate(ref+.01*ref,ref,template)["target_met"]
