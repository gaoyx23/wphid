import json
import numpy as np
import pytest
from wphid.observed_spatial_reconstruction import reconstruct_observed_stations
from experiments.audit_observed_station_reconstruction import summarize,OUT


@pytest.mark.parametrize("method",["linear","pchip"])
def test_preserves_stations_and_monotone_nonnegative_range(method):
    s=np.array([0.,.2,.5,1.]);values=np.broadcast_to(s[None,:,None],(3,4,2)).copy()
    target=np.sort(np.unique(np.r_[np.linspace(0,1,21),s]))
    output=reconstruct_observed_stations(stations=s,values=values,target_stations=target,method=method)
    for i,p in enumerate(s):np.testing.assert_array_equal(output[:,target==p,:],values[:,i:i+1,:])
    assert output.min()>=0 and output.max()<=1


def test_withheld_value_cannot_influence_prediction():
    s=np.array([0.,.2,.5,1.]);values=np.arange(4.)[None,:,None];keep=np.array([0,2,3])
    result=reconstruct_observed_stations(stations=s[keep],values=values[:,keep],target_stations=s[1:2],method="pchip")
    values[:,1]=1e12
    second=reconstruct_observed_stations(stations=s[keep],values=values[:,keep],target_stations=s[1:2],method="pchip")
    np.testing.assert_array_equal(result,second)


def test_spatial_extrapolation_and_unknown_method_rejected():
    kwargs=dict(stations=[0.,1.],values=np.ones((2,2,2)),target_stations=[1.1],method="linear")
    with pytest.raises(ValueError):reconstruct_observed_stations(**kwargs)
    kwargs.update(target_stations=[.5],method="cubic")
    with pytest.raises(ValueError):reconstruct_observed_stations(**kwargs)


def test_noise_ratio_not_reported_as_confidence():
    report=summarize(np.ones((3,1,2)),np.ones((3,1,2))*1.2,.1)
    np.testing.assert_allclose(report["rmse_in_declared_raw_noise_sigmas"],2.)
    assert not report["confidence_coverage_claim"]


def test_saved_station_audit_is_truth_blind_and_not_recovery_confirmation():
    data=json.loads((OUT/"audit.json").read_text())
    assert data["complete"] and not data["truth_states_read"]
    assert data["new_model_fits"]==data["new_weak_projections"]==0
    assert data["method_adopted"] is None
    for row in data["records"].values():
        assert set(row["held_internal_station_folds"])=={"5","15"}
        for fields in row["full_reconstruction_consistency"].values():
            assert all(field["station_values_bitwise_preserved"] for field in fields.values())
