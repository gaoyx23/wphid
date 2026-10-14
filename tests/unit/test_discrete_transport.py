import numpy as np
import pytest

from experiments.audit_discrete_transport_manufactured import manufactured_inputs, refinement
from wphid.weak.discrete_transport import project_discrete_transport_candidate


@pytest.mark.parametrize("law",["Darcy","Forchheimer"])
@pytest.mark.parametrize("ratio",[1.,3.])
def test_independent_analytic_source_converges_under_spatial_refinement(law,ratio):
    result = refinement(law,ratio)
    assert result[-1]["relative_absolute_residual"] < .1*result[0]["relative_absolute_residual"]
    assert all(r["observed_refinement_order"]>1.5 for r in result[1:])


def test_hidden_fields_and_misaligned_geometry_rejected():
    kwargs = manufactured_inputs(12,"Darcy",1.)
    kwargs["field_provenance"]["carbon"] = "truth"
    with pytest.raises(ValueError,match="hidden truth"):
        project_discrete_transport_candidate(**kwargs)
    kwargs = manufactured_inputs(12,"Darcy",1.)
    kwargs["geometry"]["distance_s"] *= 2
    with pytest.raises(ValueError,match="distances"):
        project_discrete_transport_candidate(**kwargs)


def test_sparse_or_nonfinite_observations_and_invalid_geometry_rejected():
    kwargs = manufactured_inputs(12,"Darcy",1.)
    kwargs["observed_fields"]["pO2"] = kwargs["observed_fields"]["pO2"][:,:,0]
    with pytest.raises(ValueError,match="volumetric"):
        project_discrete_transport_candidate(**kwargs)
    kwargs = manufactured_inputs(12,"Darcy",1.)
    kwargs["geometry"]["volume"] = np.zeros((12,12))
    with pytest.raises(ValueError,match="geometry"):
        project_discrete_transport_candidate(**kwargs)


def test_no_selection_or_parameter_claim_is_returned():
    result = project_discrete_transport_candidate(**manufactured_inputs(12,"Darcy",1.))
    assert result["matrix"].shape==(1,2)
    assert result["known_parameters_not_estimated"]
    assert not result["boundary_exchange_identified"]
    assert not result["calibrated_selection"]


@pytest.mark.parametrize("law,alpha",[("Darcy",0.),("Forchheimer",2.)])
@pytest.mark.parametrize("modulation",["constant","combined_opening"])
def test_face_pairing_matches_original_forward_volume_pairing(law,alpha,modulation):
    from scipy.integrate import simpson
    from wphid.transport_mechanisms import TransportCandidate,modulation_fraction
    from wphid.truth.reusable_nozzle import NozzleGeometry,NozzleGrid,NozzleTransport,_shell_mesh,_shell_diffusion
    from wphid.truth.reusable_cmc import CMCMechanics
    from wphid.weak.bell_shell import _bump
    mesh = _shell_mesh(NozzleGeometry(),NozzleGrid(meridional_cells=11,thickness_cells=8))
    t = np.linspace(0,1,31); shape=(31,11,8)
    fields = {"T":np.full(shape,1100.),"carbon":np.full(shape,.8),
        "crack":np.full(shape,.2),"stress":np.full(shape,38.e6),
        "pO2":np.broadcast_to(np.arange(88,dtype=float).reshape(1,11,8)**2,shape)}
    candidate = TransportCandidate(modulation,law,3.,alpha)
    transport,mechanics = NozzleTransport(),CMCMechanics()
    geometry=dict(volume=mesh["volume"],area_s=mesh["radius_s_face"][1:-1]*mesh["dx"],
        area_x=mesh["radius_x_face"][:,1:-1]*mesh["ds"][:,None],
        distance_s=np.diff(mesh["s"]),distance_x=np.diff(mesh["x"]))
    result = project_discrete_transport_candidate(candidate=candidate,transport=transport,
        mechanics=mechanics,species="pO2",observed_fields=fields,
        field_provenance={n:"measured" for n in fields},t=t,s=mesh["s"],x=mesh["x"],
        geometry=geometry,forcing=np.zeros(shape),window_shape=shape,stride_shape=shape)
    fraction = modulation_fraction(modulation,temperature=fields["T"],carbon=fields["carbon"],
        crack=fields["crack"],stress=fields["stress"],transport=transport,mechanics=mechanics)
    d = transport.oxygen_diffusivity_closed*(1+(transport.open_to_closed_diffusivity_ratio-1)*fraction)
    divergence = np.stack([_shell_diffusion(p,3*c,c,mesh,forchheimer_inverse_flux_scale=alpha)
                           for p,c in zip(fields["pO2"],d)])
    bt,_,_=_bump(t);bs,_,_=_bump(mesh["s"]);bx,_,_=_bump(mesh["x"])
    reference=simpson(np.sum(divergence*mesh["volume"]*bs[None,:,None]*bx[None,None,:],axis=(1,2))*bt,x=t)
    assert np.sum(result["matrix"]) == pytest.approx(reference,rel=1.e-12,abs=1.e-14)
