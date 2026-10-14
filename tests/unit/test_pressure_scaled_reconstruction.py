import json
import numpy as np
import pytest
from wphid.pressure_scaled_reconstruction import reconstruct_pressure_scaled
from experiments.run_pressure_scaled_spatial_development import OUT


def test_handles_zero_and_preserves_observations_without_pressure_floor():
    values=np.array([0.,1.,1e6])[None,:,None];s=np.array([0.,.5,1.]);target=np.linspace(0,1,5)
    result=reconstruct_pressure_scaled(stations=s,values=values,target_stations=target,noise_sigma=.1)
    np.testing.assert_array_equal(result[:,::2,:],values)
    assert result.min()==0 and result.max()==1e6
    assert np.all(np.diff(result[0,:,0])>=0)


def test_pressure_unit_change_with_sigma_is_equivariant():
    kwargs=dict(stations=[0.,1.],values=np.array([2.,1e6])[None,:,None],target_stations=[.2,.7],noise_sigma=1.)
    first=reconstruct_pressure_scaled(**kwargs)
    second=reconstruct_pressure_scaled(**{**kwargs,"values":kwargs["values"]*.001,"noise_sigma":.001})
    np.testing.assert_allclose(first*.001,second,rtol=1e-13)


def test_unknown_noise_scale_and_extrapolation_rejected():
    kwargs=dict(stations=[0.,1.],values=np.ones((2,2,3)),target_stations=[.5],noise_sigma=0.)
    with pytest.raises(ValueError):reconstruct_pressure_scaled(**kwargs)
    kwargs.update(noise_sigma=.1,target_stations=[1.1])
    with pytest.raises(ValueError):reconstruct_pressure_scaled(**kwargs)


def test_saved_pressure_proposal_is_development_only_and_not_adopted():
    data=json.loads((OUT/"audit.json").read_text())
    assert data["complete"] and not data["truth_states_read"]
    assert not data["main_method_adopted"] and not data["candidate_selection_run"]
    assert data["new_model_fits"]==data["new_weak_projections"]==0
    for row in data["records"].values():
        assert all(field["station_values_bitwise_preserved"] for field in row["full_consistency"].values())
