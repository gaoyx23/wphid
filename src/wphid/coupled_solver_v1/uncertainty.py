"""Actual raw-error propagation at a saved fit, separate from fitting metric."""

import numpy as np
from scipy.sparse import csr_matrix, vstack

from wphid.coupled_balance_v1.assembler import FIELDS, PRESSURES, TransportShape, WeakSystem
from wphid.coupled_balance_v1.noise import SharedObservationLinearization, observation_factor
from wphid.coupled_balance_v1.operators import SIDES
from wphid.coupled_candidates_v1 import gas_modulation, thermal_modulation, exponential_barrier, assemble_candidate, RateSpec
from wphid.coupled_candidates_v1.model import _rate
from wphid.coupled_observation_v1 import ObservationLayout, ObservedValue, ObservedBoundary, SparseWeakMaps
from .model import active_paths, source_terms


def raw_at_fit(problem,candidate,point,field_sigma,load_sigma):
    shape=problem.projector.state_shape;nt,ns,nx=shape
    loadnames=tuple(problem.raw_loads)
    layout=ObservationLayout([(k,shape) for k in (*FIELDS,"crack","stress")]+[("load."+k,(nt,ns)) for k in loadnames])
    fields={k:layout.observed(k,problem.fields[k]) for k in (*FIELDS,"crack","stress")}
    loads={k:layout.observed("load."+k,v) for k,v in problem.raw_loads.items()}
    maps=SparseWeakMaps(problem.projector)
    gp={k.removeprefix("gas.modulation."):v for k,v in point.items() if k.startswith("gas.modulation.")}
    tp={k.removeprefix("thermal.modulation."):v for k,v in point.items() if k.startswith("thermal.modulation.")}
    gas=gas_modulation(candidate.gas,fields,gp);thermal=thermal_modulation(candidate.thermal,fields,tp)
    external={("pO2","inner"):loads["internal_oxygen_pressure"],("pH2O","inner"):loads["external_steam_pressure"],
        ("pProducts","inner"):ObservedValue.fixed(np.zeros((nt,ns)),layout.size),("pO2","outer"):loads["external_oxygen_pressure"]}
    coverage=ObservedValue(problem.projector.trace(problem.fields["crack"],"inner"),maps.trace["inner"]@fields["crack"].jacobian)
    barrier=exponential_barrier(coverage,point["boundary.barrier_strength"])
    barriers={k:(barrier.observed_value() if k[1]=="inner" else ObservedValue.fixed(np.ones((nt,ns)),layout.size)) for k in external}
    boundary=ObservedBoundary(external,barriers,frozenset((k,s) for k in PRESSURES for s in SIDES if (k,s) not in external),
        {"inner":loads["surface_heat_flux"]},{"outer":ObservedValue.fixed(np.full((nt,ns),300.),layout.size)},frozenset(("inlet","outlet")))
    transport=TransportShape(point["gas.direction_ratio"],point["gas.steam_ratio"],point["gas.product_ratio"],point.get("gas.forchheimer_product",0.))
    independent=candidate.source=="both_independent"
    specs={} if independent else {path:RateSpec(1e5*point[f"kinetic.{path}.energy_1e5"],point[f"kinetic.{path}.pressure_order"],point["kinetic.carbon_order"]) for path in active_paths(candidate.source)}
    model,lin,report=assemble_candidate(problem.projector,layout,{k:fields[k] for k in FIELDS},
        source_candidate="null" if independent else candidate.source,rate_specs=specs,
        relative_rate=point.get("reaction.relative_rate"),shared_carbon_order=True,transport=transport,flux_law=candidate.flux,
        gas=gas,thermal=thermal,thermal_direction_ratio=point["thermal.direction_ratio"],heat_knots=problem.heat_knots,
        boundary=boundary,barrier_parameters={k:barrier.parameters for k in external if k[1]=="inner"},maps=maps)
    expected=problem.assemble(candidate,point)
    if independent:
        nrow=problem.block_rows;J=[]
        for field,path,term in source_terms(candidate.source):
            prefix=f"kinetic.{term}."
            spec=RateSpec(1e5*point[prefix+"energy_1e5"],point[prefix+"pressure_order"],point[prefix+"carbon_order"])
            expr,_=_rate(path,spec,{k:fields[k] for k in FIELDS},maps,False)
            parts=[csr_matrix((nrow,layout.size)) for _ in FIELDS]
            sign=-1 if field in ("carbon","pO2","pH2O") else 1
            parts[FIELDS.index(field)]=sign*(maps.source@expr.jacobian)
            J.append(vstack(parts,format="csr"))
        lin=SharedObservationLinearization(lin.target_jacobian,tuple(J)+lin.column_jacobians,layout.coordinate_keys)
    else:
        translate={"carbon.oxygen_rate_scale":"carbon.oxygen_response","carbon.steam_rate_scale":"carbon.steam_response",
            "oxygen_response":"pO2.oxygen_response","steam_response":"pH2O.steam_response",
            "product.oxygen_response":"pProducts.oxygen_response","product.steam_response":"pProducts.steam_response",
            "heat.oxygen_response":"T.oxygen_response","heat.steam_response":"T.steam_response"}
        actual=tuple(translate.get(k,k) for k in model.column_keys) if candidate.source=="both_free" else model.column_keys
        if actual!=expected.column_keys:raise ValueError("raw/value coefficient identity mismatch")
        np.testing.assert_allclose(expected.matrix,model.matrix,rtol=1e-12,atol=1e-12)
    variances=np.concatenate([field_sigma[k].ravel()**2 for k in (*FIELDS,"crack","stress")]+[load_sigma[k].ravel()**2 for k in loadnames])
    return expected,lin,observation_factor(variances),report
