import numpy as np
import pytest
from wphid.weak.sensor_time import project_sensor_time_weak


def test_local_decay_sign_accuracy_and_all_locations_retained():
    t=np.linspace(0,4,401)
    rates=np.array([[.2,.7],[1.,1.4]])
    u=np.exp(-t[:,None,None]*rates[None,:,:])
    matrix,target=project_sensor_time_weak(t=t,u=u,features=u[...,None],window=81,stride=40)
    # Window-major, then each observed location; no spatial average.
    assert matrix.shape==(9*4,1)
    expected=np.tile(-rates.ravel(),9)
    np.testing.assert_allclose(target/matrix[:,0],expected,rtol=2e-5)


def test_constant_baseline_is_exactly_removed_on_irregular_times():
    t=np.array([0,.09,.2,.32,.43,.51,.65,.72,.83,.93,1.])
    u=np.ones((len(t),3))*1e6
    matrix,target=project_sensor_time_weak(t=t,u=u,features=np.ones(u.shape+(1,)),window=11,stride=1)
    assert np.all(target==0)
    assert np.all(matrix>0)


def test_other_sensor_does_not_change_first_sensor_rows():
    t=np.linspace(0,2,101)
    u=np.column_stack([np.exp(-t),np.exp(-2*t)])
    features=u[...,None]
    a,b=project_sensor_time_weak(t=t,u=u,features=features,window=31,stride=15)
    altered=u.copy();altered[:,1]*=100
    aa,bb=project_sensor_time_weak(t=t,u=altered,features=altered[...,None],window=31,stride=15)
    np.testing.assert_array_equal(a[::2],aa[::2])
    np.testing.assert_array_equal(b[::2],bb[::2])


def test_bad_times_and_misaligned_features_rejected():
    t=np.linspace(0,1,11);u=np.ones((11,2))
    with pytest.raises(ValueError):project_sensor_time_weak(t=t[::-1],u=u,features=u[...,None],window=5,stride=1)
    with pytest.raises(ValueError):project_sensor_time_weak(t=t,u=u,features=np.ones((11,3,1)),window=5,stride=1)
