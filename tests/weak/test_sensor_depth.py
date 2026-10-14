import numpy as np
import pytest
from wphid.weak.sensor_depth import project_sensor_depth_weak


def inputs():
    t=np.linspace(0,1,61);x=np.linspace(.1,.8,8)
    u=np.broadcast_to((x*x)[None,None,:],(61,2,8)).copy()
    return dict(t=t,x=x,u=u,diffusivity_features=np.ones(u.shape+(1,)),
        reaction_features=np.zeros(u.shape+(1,)),volume=np.ones((2,8)),
        area_x=np.ones((2,7)),distance_x=np.diff(x),window=31,stride=15)


def test_depth_flux_sign_and_all_stations_preserved():
    kwargs=inputs();a,b=project_sensor_depth_weak(**kwargs)
    assert a.shape==(6,2) and b.shape==(6,)
    assert np.all(a[:,0]>0) # positive curvature -> positive weak diffusion
    assert np.all(b==0) # time-constant profile
    np.testing.assert_allclose(a[:,1],0,atol=1e-45)


def test_boundary_source_weights_vanish_without_boundary_identification():
    kwargs=inputs();reaction=kwargs["reaction_features"]
    reaction[:,:,(0,-1),0]=1e8
    a,_=project_sensor_depth_weak(**kwargs)
    np.testing.assert_allclose(a[:,1],0,atol=1e-45)


def test_station_rows_do_not_mix_or_reconstruct_neighbors():
    kwargs=inputs();a,b=project_sensor_depth_weak(**kwargs)
    kwargs["u"][:,1,:]*=20
    aa,bb=project_sensor_depth_weak(**kwargs)
    np.testing.assert_array_equal(a[::2],aa[::2]);np.testing.assert_array_equal(b[::2],bb[::2])


def test_inconsistent_declared_geometry_rejected():
    kwargs=inputs();kwargs["distance_x"]*=2
    with pytest.raises(ValueError):project_sensor_depth_weak(**kwargs)
