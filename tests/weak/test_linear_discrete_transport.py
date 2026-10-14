import numpy as np
import pytest
from wphid.weak.linear_discrete_transport import project_linear_discrete_transport
from wphid.weak.discrete_transport import project_discrete_transport_candidate
from wphid.transport_mechanisms import TransportCandidate,modulation_fraction
from wphid.truth.reusable_nozzle import NozzleTransport
from wphid.truth.reusable_cmc import CMCMechanics


def fixture():
    t=np.linspace(0,1,31);s=np.linspace(.1,.9,11);x=np.linspace(.1,.8,8)
    tt,ss,xx=np.meshgrid(t,s,x,indexing="ij")
    pressure=tt*.2+ss**2+xx**2
    fields=dict(T=1400+tt*10,stress=1e7*(1+ss),carbon=1-.1*tt,crack=.03*xx,pO2=pressure)
    geometry=dict(volume=np.ones((11,8)),area_s=np.ones((10,8)),area_x=np.ones((11,7)),
        distance_s=np.diff(s),distance_x=np.diff(x),meridional_face_metric=np.ones((10,8))*1.2)
    return t,s,x,fields,geometry


def test_two_direction_linear_dictionary_matches_qualified_candidate_operator():
    t,s,x,fields,geometry=fixture();transport=NozzleTransport();mechanics=CMCMechanics()
    fraction=modulation_fraction("combined_opening",temperature=fields["T"],stress=fields["stress"],
        carbon=fields["carbon"],crack=fields["crack"],transport=transport,mechanics=mechanics)
    reactions=np.stack((np.ones_like(fraction),fields["carbon"]),axis=-1)
    gains=np.array([2.,3.]);forcing=reactions@gains
    matrix,target=project_linear_discrete_transport(t=t,s=s,x=x,u=fields["pO2"],
        diffusivity_features=np.stack((np.ones_like(fraction),fraction),axis=-1),reaction_features=reactions,
        geometry=geometry,window_shape=(31,11,8),stride_shape=(15,10,8))
    old=project_discrete_transport_candidate(candidate=TransportCandidate("combined_opening","Darcy"),
        transport=transport,mechanics=mechanics,species="pO2",observed_fields=fields,
        field_provenance={k:"measured" for k in fields},t=t,s=s,x=x,geometry=geometry,forcing=forcing,
        window_shape=(31,11,8),stride_shape=(15,10,8))
    d0=transport.oxygen_diffusivity_closed
    coefficients=np.array([d0,d0*(transport.open_to_closed_diffusivity_ratio-1)])
    np.testing.assert_allclose(matrix[:,:2]@coefficients,np.sum(old["matrix"],axis=1),rtol=1e-13,atol=1e-18)
    np.testing.assert_allclose(target-matrix[:,2:]@gains,old["target"],rtol=1e-13,atol=1e-15)


def test_boundary_sources_not_used_and_bad_geometry_rejected():
    t,s,x,fields,geometry=fixture();source=np.zeros(fields["pO2"].shape+(1,))
    source[:,(0,-1),:,0]=1e9;source[:,:,(0,-1),0]=1e9
    kwargs=dict(t=t,s=s,x=x,u=fields["pO2"],diffusivity_features=np.ones_like(source),reaction_features=source,
        geometry=geometry,window_shape=(31,11,8),stride_shape=(15,10,8))
    matrix,_=project_linear_discrete_transport(**kwargs)
    np.testing.assert_allclose(matrix[:,1],0,atol=1e-45)
    geometry["distance_s"]*=2
    with pytest.raises(ValueError):project_linear_discrete_transport(**kwargs)
