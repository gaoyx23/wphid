from copy import deepcopy

import numpy as np
import pytest

from experiments.audit_discrete_transport_manufactured import manufactured_inputs
from wphid.weak.discrete_transport import project_discrete_transport_candidate
from wphid.weak.pressure_uncertainty import (
    PRESSURES,discrete_pressure_jacobian,propagate_independent_pressure_noise,
)


@pytest.mark.parametrize("law",["Darcy","Forchheimer"])
@pytest.mark.parametrize("metric",[1.,1.2])
def test_analytic_pressure_derivative_matches_directional_finite_difference(law,metric):
    kwargs=manufactured_inputs(7,law,3.)
    shape=kwargs["observed_fields"]["pO2"].shape
    kwargs["geometry"]["meridional_face_metric"]=np.full((6,7),metric)
    derivatives={n:np.full(shape,v) for n,v in zip(PRESSURES,(.2,-.1,.05))}
    for n in PRESSURES[1:]:kwargs["observed_fields"][n]=np.full(shape,2.)
    kwargs["forcing"]=sum(derivatives[n]*kwargs["observed_fields"][n] for n in PRESSURES)
    output=discrete_pressure_jacobian(forcing_pressure_derivatives=derivatives,**kwargs)
    rng=np.random.default_rng(81)
    directions={n:rng.normal(size=shape) for n in PRESSURES}
    # Local software derivative check, not a randomized scientific noise panel.
    results=[]
    for step in (1.e-5,-1.e-5):
        changed=deepcopy(kwargs)
        for n in PRESSURES:changed["observed_fields"][n]=kwargs["observed_fields"][n]+step*directions[n]
        changed["forcing"]=sum(derivatives[n]*changed["observed_fields"][n] for n in PRESSURES)
        results.append(project_discrete_transport_candidate(**changed)["residual"])
    predicted=sum(np.sum(output["pressure_jacobians"][n]*directions[n][None,:,:,:],axis=(1,2,3)) for n in PRESSURES)
    np.testing.assert_allclose(predicted,(results[0]-results[1])/2.e-5,rtol=2.e-7,atol=1.e-9)


def test_covariance_matches_explicit_diagonal_matrix_and_retains_cross_channel_correlation():
    jacobians={n:np.arange(16,dtype=float).reshape(2,2,2,2)*(i+1) for i,n in enumerate(PRESSURES)}
    scales={n:np.full((2,2,2),.1*(i+1)) for i,n in enumerate(PRESSURES)}
    result=propagate_independent_pressure_noise(pressure_jacobians=jacobians,standard_deviations=scales)
    j=np.concatenate([jacobians[n].reshape(2,-1) for n in PRESSURES],axis=1)
    variance=np.concatenate([scales[n].ravel()**2 for n in PRESSURES])
    np.testing.assert_allclose(result,j@np.diag(variance)@j.T)
    assert result[0,1]>0


def test_missing_source_derivative_is_rejected():
    kwargs=manufactured_inputs(7,"Darcy",1.)
    with pytest.raises(ValueError,match="every pressure"):
        discrete_pressure_jacobian(forcing_pressure_derivatives={},**kwargs)


def test_negative_noise_scale_is_rejected():
    j={n:np.ones((2,2,2,2)) for n in PRESSURES}
    std={n:np.ones((2,2,2)) for n in PRESSURES};std["pO2"]*=-1
    with pytest.raises(ValueError,match="nonnegative"):
        propagate_independent_pressure_noise(pressure_jacobians=j,standard_deviations=std)
