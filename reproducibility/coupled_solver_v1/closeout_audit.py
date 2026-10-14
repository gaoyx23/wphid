"""One saved-clean truth-point compatibility and complete receipt audit.

Oracle values are used only after all blind fits, never to choose a model,
parameter, weighting or budget. No physical integration or fit is repeated.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]


def main():
    from reproducibility.coupled_solver_v1.workflow import PROTOCOL,sha,write,verify_freeze,problems_from_public
    from wphid.coupled_solver_v1 import Candidate,ValueProblem,parameter_box
    from wphid.coupled_balance_v1.assembler import BoundaryInputs,PRESSURES,FIELDS
    from wphid.coupled_balance_v1.operators import SIDES,Modulation
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"));output=ROOT/protocol["output_root"]
    verify_freeze(output,protocol)
    if (output/"closeout-audit.json").exists():raise SystemExit("completed closeout retained; no repeated checks")
    receipt=json.loads((output/"execution-complete.json").read_text(encoding="utf-8"))
    if sha(output/"assessment.json")!=receipt["assessment_sha256"]:raise SystemExit("scientific assessment changed")
    fitreceipt=json.loads((output/"fits-complete.json").read_text(encoding="utf-8"))
    if any(sha(output/"fits"/i["path"])!=i["sha256"] for i in fitreceipt["files"]):raise SystemExit("fit changed")
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"))
    if any(sha(output/i["path"])!=i["sha256"] for i in index["files"]):raise SystemExit("input changed")
    truth=json.loads((output/"truth_scoring_only/truth.json").read_text(encoding="utf-8"));compatibility=[]
    archive=output/"truth_scoring_only/recovered_clean_trajectory_archive_r1"
    for case,role in enumerate(protocol["data"]["source_roles"]):
        block=next(b for b in index["blocks"] if b["identity"]==f"block_{case:02d}_00")
        public,_,_=problems_from_public(output,block,index)
        target=truth[block["identity"]];candidate=Candidate(target["source"])
        point={k:target["parameters"][k] for k in parameter_box(candidate)}
        for mission,p in enumerate(public):
            with np.load(archive/f"case_{case:02d}_mission_{mission:02d}.npz",allow_pickle=False) as saved:
                fields={k:saved["state."+k].copy() for k in (*FIELDS,"crack","stress")}
                loads={k:saved["load."+k].copy() for k in ("internal_oxygen_pressure","external_steam_pressure","external_oxygen_pressure","surface_heat_flux")}
            nt,ns,nx=p.projector.state_shape
            ext={("pO2","inner"):loads["internal_oxygen_pressure"],("pH2O","inner"):loads["external_steam_pressure"],("pProducts","inner"):np.zeros((nt,ns)),("pO2","outer"):loads["external_oxygen_pressure"]}
            boundary=BoundaryInputs(ext,{k:Modulation(np.ones((nt,ns))) for k in ext},frozenset((f,s) for f in PRESSURES for s in SIDES if (f,s) not in ext),
                {"inner":loads["surface_heat_flux"]},{"outer":np.full((nt,ns),300.)},frozenset(("inlet","outlet")))
            clean=ValueProblem(p.projector,fields,boundary)
            model=clean.assemble(candidate,point);a=np.array([target["coefficients"][k] for k in model.column_keys])
            prediction=model.matrix@a;residual=model.target-prediction;n=clean.block_rows
            values={}
            for j,k in enumerate(FIELDS):
                r=residual[j*n:(j+1)*n];b=model.target[j*n:(j+1)*n];pred=prediction[j*n:(j+1)*n]
                denominator=max(float(np.linalg.norm(b)),float(np.linalg.norm(pred)))
                values[k]={"absolute_residual_norm":float(np.linalg.norm(r)),"symmetric_relative_residual":float(np.linalg.norm(r)/denominator) if denominator>1e-12 else None,
                    "zero_balance_relative_error_not_defined":denominator<=1e-12}
            compatibility.append({"case":case,"role":role,"mission":mission,"weak_balance":values})
    program_stats={}
    for program in ("wphid","adam"):
        fits=[json.loads((output/"fits"/i["path"]).read_text(encoding="utf-8"))["fit"] for i in fitreceipt["files"] if i["path"].endswith("_"+program+".json")]
        cpu=[f.get("cpu_seconds",0.) for f in fits]
        truncated=sum(any(r.get("budget_exhausted",False) or r.get("status")=="budget_exhausted" for r in f.get("restarts",[])) for f in fits)
        errors=sum(bool(f.get("errors")) for f in fits)
        program_stats[program]={"attempts":len(fits),"finite_fits":sum(f["fitted"] for f in fits),"solver_cpu_seconds_sum":sum(cpu),
            "solver_cpu_seconds_median":float(np.median(cpu)),"maximum_solver_cpu_seconds":max(cpu),"budget_truncated_fits":truncated,
            "fits_with_retained_restart_errors":errors,"preprocessing_and_process_overhead_included_in_solver_cpu":False,
            "cpu_cap_overshoot_is_reported":True}
    qualified=[]
    for folder in ("paper_coupled_declaration_v1/candidates/software_qualification_20261013_attempt01",
                   "paper_coupled_declaration_v1/observation/software_qualification_20261013_attempt01",
                   "paper_coupled_solver_v1/software/qualification_20261013_attempt01",
                   "paper_coupled_solver_v1/workflow_software/qualification_20261013_attempt01",
                   "paper_coupled_solver_v1/postprocess_software/qualification_20261013_attempt01",
                   "paper_coupled_solver_v1/noise_replay_r1_software/qualification_20261013_attempt01"):
        directory=ROOT/"artifacts"/folder;rec=json.loads((directory/"execution-complete.json").read_text(encoding="utf-8"));freeze=json.loads((directory/"source-freeze.json").read_text(encoding="utf-8"))
        current=all(sha(ROOT/i["path"])==i["sha256"] for i in freeze["sources"])
        matched=sha(directory/"assessment.json")==rec["assessment_sha256"] and sha(directory/"source-freeze.json")==rec["source_freeze_sha256"]
        audit=json.loads((directory/"assessment.json").read_text(encoding="utf-8"))
        qualified.append({"path":folder,"passed":bool(rec["software_passed"] and current and matched),"checks":audit["artificial_checks_passed"]})
    result={"status":"completed","science_freeze_and_receipts_valid":True,"all_63_current_artificial_checks":all(i["passed"] for i in qualified),
        "qualification_records":qualified,"program_statistics":program_stats,"saved_clean_truth_point_compatibility":compatibility,
        "truth_point_diagnostic_not_parameter_recovery":True,"actual_error_covariance_not_the_fixed_fitting_metric":True,
        "new_fits":0,"new_trajectories":0,"new_noise_blocks":0,"original_eight_package_count":"6/8", "source_sha256":sha(Path(__file__))}
    write(output/"closeout-audit.json",result)
    print(json.dumps({"all_current_qualification_records_valid":result["all_63_current_artificial_checks"],"program_statistics":program_stats,
        "oracle_relative_residual_ranges":{k:[min(vals),max(vals)] for k in FIELDS if (vals:=[r["weak_balance"][k]["symmetric_relative_residual"] for r in compatibility if r["weak_balance"][k]["symmetric_relative_residual"] is not None])}},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
