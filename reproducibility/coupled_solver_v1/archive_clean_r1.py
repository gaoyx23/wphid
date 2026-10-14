"""Archive replay R1: compare each channel in its own units before taking max.

The old archival script is preserved and has not been executed on scientific
data. Its scalar defect accumulation could rescale the previous channel's
defect. This R1 changes that reference bookkeeping only, not noise inversion.
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from reproducibility.coupled_solver_v1.archive_clean_from_noise_replay import recover,FIELD_ORDER,LOAD_ORDER


def compare_recovered(first,second):
    return max(float(np.max(np.abs(a[k]-b[k])))/max(float(np.max(np.abs(a[k]))),1.)
        for a,b in zip(first[:2],second[:2]) for k in a)


def main():
    from reproducibility.coupled_solver_v1.workflow import sha,write,verify_freeze,PROTOCOL
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"));output=ROOT/p["output_root"];verify_freeze(output,p)
    if not (output/"fits-complete.json").exists():raise SystemExit("archive only after all fits complete")
    archive=output/"truth_scoring_only/recovered_clean_trajectory_archive_r1"
    if archive.exists():raise SystemExit("completed archive retained; no repeat")
    archive.mkdir(exist_ok=False)
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"));blocks={i["identity"]:i for i in index["blocks"]}
    records=[]
    for case in range(len(p["data"]["source_roles"])):
        repeats=[]
        for r in (0,1):
            block=blocks[f"block_{case:02d}_{r:02d}"]
            if sha(output/block["path"])!=block["sha256"]:raise SystemExit("saved input changed")
            with np.load(output/block["path"],allow_pickle=False) as data:
                trajectories=[]
                for mission in range(p["data"]["mission_count"]):
                    obs={k:data[f"m{mission}.state.{k}"].copy() for k in FIELD_ORDER}
                    loads={k:data[f"m{mission}.load.{k}"].copy() for k in LOAD_ORDER}
                    seed=p["data"]["noise_seed_root"]+case*100+r*10+mission
                    clean,cleanloads=recover(obs,loads,seed,p["observation_contract"])
                    trajectories.append((clean,cleanloads,data["t"].copy()))
                repeats.append(trajectories)
        for mission in range(p["data"]["mission_count"]):
            first,second=repeats[0][mission],repeats[1][mission];defect=compare_recovered(first,second)
            if defect>64*np.finfo(float).eps:raise SystemExit("saved streams disagree, no regeneration")
            name=f"case_{case:02d}_mission_{mission:02d}.npz"
            arrays={"t":first[2],**{"state."+k:v for k,v in first[0].items()},**{"load."+k:v for k,v in first[1].items()}}
            with (archive/name).open("xb") as f:np.savez_compressed(f,**arrays)
            records.append({"path":name,"sha256":sha(archive/name),"two_stream_relative_defect":defect})
    write(archive/"assessment.json",{"status":"completed","source_sha256":sha(Path(__file__)),"inversion_source_sha256":sha(ROOT/"reproducibility/coupled_solver_v1/archive_clean_from_noise_replay.py"),
        "files":records,"maximum_relative_defect":max(i["two_stream_relative_defect"] for i in records),
        "original_clean_byte_identity_not_claimed":True,"new_integrations":0,"new_noise_realizations":0,"new_fits":0,"used_for_estimation":False})
    write(archive/"execution-complete.json",{"status":"completed","assessment_sha256":sha(archive/"assessment.json"),"trajectory_snapshot_count":len(records)})
    print(json.dumps({"archived_trajectory_snapshots":len(records),"maximum_relative_defect":max(i["two_stream_relative_defect"] for i in records),"new_integrations":0}))


if __name__=="__main__":main()
