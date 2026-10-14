"""New controlled continuous finite-volume data; never repairs old arrays.

The generator does not call the inverse weak assembler or add manufactured
forcing. Crack/stress are prescribed observed drivers, not recovered laws.
Truth coordinates are returned to the sealed scorer only.
"""

import numpy as np
from scipy.integrate import solve_ivp


GEOMETRY = {"length":.988,"exit_radius":.469,"expansion_ratio":62.5,
    "wall_thickness":.0065,"throat_angle_deg":30.,"exit_angle_deg":7.}
NAMES=("carbon","pO2","pH2O","pProducts","T")


def truth_point(source,near=False):
    p={"kinetic.oxygen.energy_1e5":1.23,"kinetic.steam.energy_1e5":1.45,
       "kinetic.oxygen.pressure_order":1.,"kinetic.steam.pressure_order":.75,
       "kinetic.carbon_order":1.,"reaction.relative_rate":1. if near else .42/.11,
       "gas.direction_ratio":1.3,"gas.steam_ratio":1.4,"gas.product_ratio":3.8/3.,
       "thermal.direction_ratio":2.,"boundary.barrier_strength":.3,
       "gas.modulation.stress_gain_per_pa":2.0e-8,"gas.modulation.stress_cap":8.5,
       "gas.modulation.damage_gain":17*5/9.6,"gas.modulation.carbon_gain":17*3/9.6,
       "gas.modulation.closure_gain":17*.45/9.6,"gas.modulation.closure_temperature":1500.,
       "gas.modulation.closure_span":450.,"gas.modulation.total_cap":17.,
       "thermal.modulation.loss":.45,"thermal.modulation.stiffness_floor":.10,
       "thermal.modulation.conductivity_floor":.35}
    rho_cp=1900.*1000.
    a={"carbon_rate_scale":.11,"oxygen_response":4e5*.11,"steam_response":7e5*.42,
       "product_response":5.2e5*.11,"heat_response":2.2e8/rho_cp*.11,
       "gas_scale":3e-7,"surface_transfer":2.5e-4,"thermal_scale":9./rho_cp,
       "heat_input_scale":1./rho_cp,"radiation_scale":.85*5.670374419e-8/rho_cp,
       "heat_capacity_contrast":.8}
    if source=="steam_only":
        a.update(carbon_rate_scale=.42,product_response=5.2e5*.42,heat_response=2.2e8/rho_cp*.42)
    if source in ("both_free","both_independent"):
        amplitudes={"oxygen":.11,"steam":.42}
        for path in ("oxygen","steam"):
            amp=amplitudes[path]
            for field in ("carbon","pO2" if path=="oxygen" else "pH2O","pProducts","T"):
                factor={"carbon":1.,"pO2":4e5,"pH2O":7e5,"pProducts":5.2e5,"T":2.2e8/rho_cp}[field]
                if source=="both_free" and field=="T" and path=="steam":factor*=.6
                if source=="both_free" and field=="pProducts" and path=="steam":factor*=1.3
                a[f"{field}.{path}_response"]=amp*factor
                if source=="both_independent":
                    shift={"carbon":0.,"pO2":.12,"pH2O":-.1,"pProducts":-.08,"T":.06}[field]
                    p[f"kinetic.{field}.{path}.energy_1e5"]=p[f"kinetic.{path}.energy_1e5"]+shift
                    p[f"kinetic.{field}.{path}.pressure_order"]=p[f"kinetic.{path}.pressure_order"]+(0.15 if field=="pProducts" else 0.)
                    p[f"kinetic.{field}.{path}.carbon_order"]={"carbon":1.,"pO2":.8,"pH2O":1.2,"pProducts":1.3,"T":.9}[field]
    return p,a


def drivers(t,ns,nx,mission):
    si=np.linspace(0.,1.,ns)[:,None];xi=np.linspace(0.,1.,nx)[None,:]
    crack=.025+.055*(1-np.exp(-t/300.))*(1+.15*si+.08*xi)
    stress=2e7+(1.3e8*(1-np.exp(-t/60.)))*(1-.35*si+.08*xi)*(1+.10*np.sin(t/160.+mission))
    return np.broadcast_to(crack,(ns,nx)).copy(),np.broadcast_to(stress,(ns,nx)).copy()


def loads(t,ns,mission,duration):
    s=np.linspace(0.,1.,ns)
    duty=(1-np.exp(-t/15.))*(1-np.exp(-(duration-t)/18.))
    heat=(575000.,700000.,500000.)[mission]*duty*(1-.65*s)
    return {"internal_oxygen_pressure":(1200.,2000.,800.)[mission]*duty*np.ones(ns),
        "external_steam_pressure":(6e5,6e5,9e5)[mission]*duty*np.ones(ns),
        "external_oxygen_pressure":np.full(ns,21000.),"surface_heat_flux":heat}


def generate(source,mesh,times,mission,*,near=False):
    ns,nx=mesh.shape;duration=float(times[-1]);p,a=truth_point(source,near)
    initial=np.stack((np.ones(mesh.shape),np.zeros(mesh.shape),np.zeros(mesh.shape),np.zeros(mesh.shape),np.full(mesh.shape,330.)))
    def rhs(t,vector):
        y=vector.reshape(5,ns,nx);C,O,W,P,T=y
        if np.any(T<=0) or np.any(C<=.02) or min(O.min(),W.min(),P.min()) < -1e-4:
            raise ValueError("continuous controlled model left its declared domain; no regenerated trajectory")
        # Only negative integration roundoff is tolerated, not observed-state
        # clipping; every negative value is subsequently counted in the receipt.
        pressure={"oxygen":np.maximum(O,0.),"steam":np.maximum(W,0.)}
        crack,stress=drivers(t,ns,nx,mission)
        stresspart=np.minimum(p["gas.modulation.stress_gain_per_pa"]*stress,p["gas.modulation.stress_cap"])
        hot=np.clip((T-1500.)/450.,0.,1.)
        opening=np.clip(stresspart+p["gas.modulation.damage_gain"]*crack+p["gas.modulation.carbon_gain"]*(1-C)-p["gas.modulation.closure_gain"]*hot,0.,17.)
        gain=1+opening
        conductivity=np.maximum(1-.45*(1-np.maximum(C*(1-crack),.1)),.35)
        external=loads(t,ns,mission,duration)
        def divergence(value,scale,direction,modulation):
            ds=-scale*direction*.5*(modulation[1:]+modulation[:-1])*np.diff(value,axis=0)/mesh.distance_s
            dx=-scale*.5*(modulation[:,1:]+modulation[:,:-1])*np.diff(value,axis=1)/mesh.distance_x
            result=np.zeros_like(value)
            result[:-1]-=ds*mesh.area_s;result[1:]+=ds*mesh.area_s
            result[:,:-1]-=dx*mesh.area_x;result[:,1:]+=dx*mesh.area_x
            return result/mesh.volume
        out=np.zeros_like(y)
        for index,key,ratio in ((1,"pO2",1.),(2,"pH2O",1.4),(3,"pProducts",3.8/3.)):
            out[index]=divergence(y[index],a["gas_scale"]*ratio,1.3,gain)
            drive=external["internal_oxygen_pressure" if index==1 else "external_steam_pressure"] if index!=3 else np.zeros(ns)
            out[index,:,0]+=a["surface_transfer"]*np.exp(-.3*crack[:,0])*(drive-y[index,:,0])*mesh.boundary_areas["inner"]/mesh.volume[:,0]
        out[1,:,-1]+=a["surface_transfer"]*(external["external_oxygen_pressure"]-O[:,-1])*mesh.boundary_areas["outer"]/mesh.volume[:,-1]
        out[4]=divergence(T,a["thermal_scale"],2.,conductivity)
        out[4,:,0]+=a["heat_input_scale"]*external["surface_heat_flux"]*mesh.boundary_areas["inner"]/mesh.volume[:,0]
        out[4,:,-1]-=a["radiation_scale"]*(T[:,-1]**4-300.**4)*mesh.boundary_areas["outer"]/mesh.volume[:,-1]
        paths=() if source=="null" else (("oxygen",) if source=="oxygen_only" else (("steam",) if source=="steam_only" else ("oxygen","steam")))
        def phi(path,field=None):
            prefix=f"kinetic.{field}.{path}." if source=="both_independent" else f"kinetic.{path}."
            q=p[prefix+"carbon_order"] if source=="both_independent" else 1.
            return np.exp(-1e5*p[prefix+"energy_1e5"]/(8.314462618*T))*(pressure[path]/(2e4 if path=="oxygen" else 1e6))**p[prefix+"pressure_order"]*C**q
        if source in ("both_free","both_independent"):
            for path in paths:
                for field in ("carbon","pO2" if path=="oxygen" else "pH2O","pProducts","T"):
                    index=NAMES.index(field);sign=-1 if index<3 else 1
                    out[index]+=sign*a[f"{field}.{path}_response"]*phi(path,field)
        elif paths:
            rates={path:phi(path) for path in paths}
            mixture=next(iter(rates.values())) if len(paths)==1 else rates["oxygen"]+p["reaction.relative_rate"]*rates["steam"]
            out[0]-=a["carbon_rate_scale"]*mixture
            if "oxygen" in rates:out[1]-=a["oxygen_response"]*rates["oxygen"]
            if "steam" in rates:out[2]-=a["steam_response"]*rates["steam"]
            product=mixture;heat=mixture
            if source=="both_oxygen_heat":heat=rates["oxygen"]
            if source=="both_swapped_product":product=p["reaction.relative_rate"]*rates["oxygen"]+rates["steam"]
            out[3]+=a["product_response"]*product;out[4]+=a["heat_response"]*heat
        fraction=np.clip((T-293.15)/(1873.15-293.15),0.,1.)
        out[4]/=1+.8*fraction
        return out.ravel()
    solution=solve_ivp(rhs,(float(times[0]),duration),initial.ravel(),t_eval=times,method="BDF",rtol=1e-8,atol=1e-9,max_step=2.)
    if not solution.success:raise ValueError(solution.message)
    values=solution.y.T.reshape(len(times),5,ns,nx)
    if np.any(values[:,:4]<-1e-7):raise ValueError("saved integration states negative beyond declared roundoff")
    result={key:values[:,i] for i,key in enumerate(NAMES)}
    D,S=zip(*(drivers(t,ns,nx,mission) for t in times));result.update(crack=np.array(D),stress=np.array(S))
    rawloads={key:np.array([loads(t,ns,mission,duration)[key] for t in times]) for key in loads(0.,ns,mission,duration)}
    return result,rawloads,p,a,{"solver_success":True,"nfev":solution.nfev,"minimum_carbon":float(result["carbon"].min()),
        "maximum_temperature":float(result["T"].max()),"negative_roundoff_count":int(np.count_nonzero(values[:,:4]<0)),
        "state_clipping":False,"manufactured_compensating_forcing":False}


def observe(fields,loads,rng,contract):
    """Bounded independent instrument errors; exact fixed first field snapshot.

Carbon noise measures carbon loss. Multiplicative pressure/driver noise has
variance zero at a true zero as a law, not as an inference from weak source use.
The contract is intentionally controlled and does not repair archived clipping.
"""
    observed={};sigma={};initial_count=0
    for key,value in fields.items():
        u=rng.uniform(-np.sqrt(3.),np.sqrt(3.),value.shape)
        if key=="T":amplitude=np.full(value.shape,contract["temperature_sigma_K"])
        elif key=="carbon":amplitude=contract["carbon_loss_relative_sigma"]*(1-value)
        else:amplitude=contract["relative_sigma"][key]*value
        error=amplitude*u;error[0]=0.
        observed[key]=value+error
        # Plug-in variance is an explicit approximation using observations only.
        if key=="T":estimate=amplitude.copy()
        elif key=="carbon":estimate=contract["carbon_loss_relative_sigma"]*np.abs(1-observed[key])/np.sqrt(1+contract["carbon_loss_relative_sigma"]**2)
        else:estimate=contract["relative_sigma"][key]*np.abs(observed[key])/np.sqrt(1+contract["relative_sigma"][key]**2)
        estimate[0]=0.;sigma[key]=estimate;initial_count+=value[0].size
    observed_loads={};load_sigma={}
    for key,value in loads.items():
        rel=contract["load_relative_sigma"]
        observed_loads[key]=value*(1+rel*rng.uniform(-np.sqrt(3.),np.sqrt(3.),value.shape))
        load_sigma[key]=rel*np.abs(observed_loads[key])/np.sqrt(1+rel**2)
    if (np.any(observed["T"]<=0) or np.any(observed["carbon"]<=.02) or np.any(observed["carbon"]>1)
        or np.any(observed["crack"]<0) or np.any(observed["crack"]>1)
        or any(np.any(observed[k]<0) for k in ("pO2","pH2O","pProducts","stress"))):
        raise ValueError("declared bounded sensor law left physical domain; no redraw or clipping")
    return observed,observed_loads,sigma,load_sigma,{"fixed_initial_coordinates":initial_count,
        "noise_distribution":"bounded_uniform_independent","clipping_or_reconstruction":False,
        "variance_source":"declared_instrument_contract_with_observation_only_plugin"}
