"""Archive new clean snapshots by inverting the saved original noise stream.

No integration, new random realization or fitting is repeated. The two stored
noise repeats independently recover the same clean trajectory up to roundoff.
Recovered arrays are sealed archival data, never inverse-estimator input.
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
FIELD_ORDER=("carbon","pO2","pH2O","pProducts","T","crack","stress")
LOAD_ORDER=("internal_oxygen_pressure","external_steam_pressure","external_oxygen_pressure","surface_heat_flux")


def recover(observed,loads,seed,contract):
    rng=np.random.default_rng(seed);clean={};cleanloads={}
    for key in FIELD_ORDER:
        value=observed[key];u=rng.uniform(-np.sqrt(3.),np.sqrt(3.),value.shape)
        if key=="T":result=value-contract["temperature_sigma_K"]*u
        elif key=="carbon":result=1-(1-value)/(1-contract["carbon_loss_relative_sigma"]*u)
        else:result=value/(1+contract["relative_sigma"][key]*u)
        result[0]=value[0];clean[key]=result
    for key in LOAD_ORDER:
        value=loads[key];u=rng.uniform(-np.sqrt(3.),np.sqrt(3.),value.shape)
        cleanloads[key]=value/(1+contract["load_relative_sigma"]*u)
    return clean,cleanloads


def main():
    from reproducibility.coupled_solver_v1.workflow import sha,write,verify_freeze,PROTOCOL
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"));output=ROOT/p["output_root"];verify_freeze(output,p)
    if not (output/"fits-complete.json").exists():raise SystemExit("archive only after all blind fits complete")
    archive=output/"truth_scoring_only/recovered_clean_trajectory_archive"
    if archive.exists():raise SystemExit("completed clean archive retained; no replay repeated")
    archive.mkdir(exist_ok=False)
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"));blocks={i["identity"]:i for i in index["blocks"]}
    records=[];maximum=0.
    for case in range(len(p["data"]["source_roles"])):
        repeat=[]
        for r in (0,1):
            block=blocks[f"block_{case:02d}_{r:02d}"]
            if sha(output/block["path"])!=block["sha256"]:raise SystemExit("saved observation identity changed")
            with np.load(output/block["path"],allow_pickle=False) as data:
                trajectories=[]
                for mission in range(p["data"]["mission_count"]):
                    observed={k:data[f"m{mission}.state.{k}"].copy() for k in FIELD_ORDER}
                    loads={k:data[f"m{mission}.load.{k}"].copy() for k in LOAD_ORDER}
                    seed=p["data"]["noise_seed_root"]+case*100+r*10+mission
                    clean,cleanloads=recover(observed,loads,seed,p["observation_contract"])
                    trajectories.append((clean,cleanloads,data["t"].copy()))
                repeat.append(trajectories)
        for mission in range(p["data"]["mission_count"]):
            first,second=repeat[0][mission],repeat[1][mission];defect=0.
            for a,b in zip(first[:2],second[:2]):
                for key in a:
                    defect=max(defect,float(np.max(np.abs(a[key]-b[key])))/max(float(np.max(np.abs(a[key]))),1.))
            if defect>64*np.finfo(float).eps:raise SystemExit("two saved noise streams disagree; no regeneration")
            maximum=max(maximum,defect);filename=f"case_{case:02d}_mission_{mission:02d}.npz"
            arrays={"t":first[2],**{"state."+k:v for k,v in first[0].items()},**{"load."+k:v for k,v in first[1].items()}}
            with (archive/filename).open("xb") as f:np.savez_compressed(f,**arrays)
            records.append({"path":filename,"sha256":sha(archive/filename),"two_stream_relative_defect":defect})
    write(archive/"assessment.json",{"status":"completed","source_sha256":sha(Path(__file__)),"files":records,
        "maximum_relative_defect":maximum,"exact_original_byte_identity_not_claimed":True,"original_roundoff_preserved_in_public_observations":True,
        "new_integrations":0,"new_noise_realizations":0,"new_fits":0,"used_for_fit_or_selection":False})
    write(archive/"execution-complete.json",{"status":"completed","assessment_sha256":sha(archive/"assessment.json"),"trajectory_snapshot_count":len(records)})
    print(json.dumps({"archived_clean_trajectory_snapshots":len(records),"two_stream_relative_defect":maximum,"new_integrations":0,"new_fits":0}))


if __name__=="__main__":main()
