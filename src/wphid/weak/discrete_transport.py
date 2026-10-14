"""Observation-facing finite-volume weak transport balances.

Geometry is supplied explicitly, not read from a truth trajectory. Boundary
cell test weights vanish: this operator does not identify boundary exchange.
It assembles known-parameter balances, not a calibrated candidate selector.
"""
import numpy as np
from scipy.integrate import simpson

from ..transport_mechanisms import modulation_fraction, constitutive_flux
from .axisymmetric import axisymmetric_window_slices
from .bell_shell import _bump


def project_discrete_transport_candidate(*, candidate, transport, mechanics,
        species, observed_fields, field_provenance, t, s, x, geometry,
        forcing, window_shape, stride_shape):
    coordinates = [np.asarray(v,dtype=float) for v in (t,s,x)]
    for v in coordinates:
        if v.ndim != 1 or len(v)<3 or not np.all(np.isfinite(v)) or np.any(np.diff(v)<=0):
            raise ValueError("finite increasing time and cell-center coordinates required")
    time, meridional, depth = coordinates
    shape = (len(time),len(meridional),len(depth))
    required = ("T","stress","carbon","crack",species)
    if any(n not in observed_fields or field_provenance.get(n) not in
           ("measured","declared_reconstruction") for n in required):
        raise ValueError("explicit measured/reconstructed rich fields required; hidden truth prohibited")
    fields = {n:np.asarray(observed_fields[n],dtype=float) for n in required}
    source = np.asarray(forcing,dtype=float)
    if any(v.shape!=shape or not np.all(np.isfinite(v)) for v in (*fields.values(),source)):
        raise ValueError("finite aligned volumetric observations and forcing required")
    expected = {"volume":shape[1:],"area_s":(shape[1]-1,shape[2]),
                "area_x":(shape[1],shape[2]-1),"distance_s":(shape[1]-1,),
                "distance_x":(shape[2]-1,)}
    mesh = {}
    for n, dims in expected.items():
        v = np.asarray(geometry.get(n),dtype=float)
        if v.shape!=dims or not np.all(np.isfinite(v)) or np.any(v<=0):
            raise ValueError("positive finite explicit geometry required: " + n)
        mesh[n] = v
    for n, coordinate in (("distance_s",meridional),("distance_x",depth)):
        if not np.allclose(mesh[n],np.diff(coordinate),rtol=1.e-10,atol=1.e-14):
            raise ValueError("face distances must match declared cell-center coordinates")
    # s is a reference meridional coordinate. A curved offset shell can have
    # a physical arc length h_s(s,x) ds different from ds. Never infer it from
    # hidden solver geometry; the observation geometry must declare it.
    metric = np.asarray(geometry.get("meridional_face_metric",
        np.ones(expected["area_s"])),dtype=float)
    if metric.shape!=expected["area_s"] or not np.all(np.isfinite(metric)) or np.any(metric<=0):
        raise ValueError("positive finite meridional face metric required")
    diffusivity_name = {"pO2":"oxygen_diffusivity_closed","pH2O":"steam_diffusivity_closed",
                        "pProducts":"product_diffusivity_closed"}.get(species)
    if diffusivity_name is None:
        raise ValueError("unsupported gas species")
    configured = candidate.configure_transport(transport)
    d0 = getattr(configured,diffusivity_name)
    amplitude = configured.open_to_closed_diffusivity_ratio
    if not np.isfinite(d0) or d0<=0 or not np.isfinite(amplitude) or amplitude<1:
        raise ValueError("positive diffusivity and open/closed ratio >=1 required")
    fraction = modulation_fraction(candidate.modulation,temperature=fields["T"],
        stress=fields["stress"],carbon=fields["carbon"],crack=fields["crack"],
        transport=configured,mechanics=mechanics)
    d = d0*(1+(amplitude-1)*fraction)
    pressure = fields[species]
    js = constitutive_flux(-configured.meridional_to_through_diffusivity_ratio*
        .5*(d[:,1:,:]+d[:,:-1,:])*np.diff(pressure,axis=1)/
        (mesh["distance_s"][None,:,None]*metric[None,:,:]),
        configured.forchheimer_inverse_flux_scale)
    jx = constitutive_flux(-.5*(d[:,:,1:]+d[:,:,:-1])*np.diff(pressure,axis=2)/
        mesh["distance_x"][None,None,:],configured.forchheimer_inverse_flux_scale)
    rows, targets = [], []
    for slices in axisymmetric_window_slices(shape,window_shape,stride_shape):
        tt, ss, xx = [c[sl] for c,sl in zip(coordinates,slices)]
        bt, bt_t, _ = _bump(tt); bs,_,_ = _bump(ss); bx,_,_ = _bump(xx)
        spatial_test = np.zeros(shape[1:])
        spatial_test[slices[1],slices[2]] = bs[:,None]*bx[None,:]
        weight = spatial_test*mesh["volume"]
        local_pressure = pressure[slices[0]]
        local_source = source[slices[0]]
        target = -simpson(np.sum(local_pressure*weight,axis=(1,2))*bt_t,x=tt)
        target -= simpson(np.sum(local_source*weight,axis=(1,2))*bt,x=tt)
        # Summation by parts transfers divergence to differences of the test.
        col_s = simpson(np.sum(js[slices[0]]*mesh["area_s"]*
            np.diff(spatial_test,axis=0),axis=(1,2))*bt,x=tt)
        col_x = simpson(np.sum(jx[slices[0]]*mesh["area_x"]*
            np.diff(spatial_test,axis=1),axis=(1,2))*bt,x=tt)
        rows.append([col_s,col_x]); targets.append(target)
    if not rows:
        raise ValueError("window configuration produced no balances")
    matrix, target = np.asarray(rows),np.asarray(targets)
    return dict(candidate_id=candidate.candidate_id,matrix=matrix,target=target,
        residual=target-np.sum(matrix,axis=1),known_parameters_not_estimated=True,
        spatial_discretization="explicit_face_flux_summation_by_parts",
        geometry_metric_mode=("declared_meridional_metric" if "meridional_face_metric" in geometry
                              else "unit_reference_metric"),
        boundary_exchange_identified=False,calibrated_selection=False)
