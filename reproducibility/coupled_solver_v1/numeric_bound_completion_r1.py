"""Finish only fault-interrupted starts inside original remaining budgets.

V1 observations, methods, completed starts, failed logs and scores stay immutable.
R1 changes inclusive unit-to-physical floating endpoints only. No extra input,
candidate, seed, objective-call ceiling, CPU ceiling, weight or precision gate.
"""

from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from reproducibility.coupled_solver_v1.workflow import sha,write,check_runtime,verify_freeze,problems_from_public
REPAIR=ROOT/"reproducibility/coupled_solver_v1/numeric-bound-completion-r1-protocol-20261013.json"


def worker(job):
    from wphid.coupled_solver_v1 import Candidate,common_metric
    from wphid.coupled_solver_v2 import fit_profile
    output=Path(job["output"]);check_runtime(output)
    old=json.loads((ROOT/job["original_file"]).read_text(encoding="utf-8"))
    contract=json.loads((output/"public/fit-contract.json").read_text(encoding="utf-8"))
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"))
    problems,_,_=problems_from_public(output,job["block"],index)
    metric=common_metric(problems,contract["reference_metric_sigma"])
    candidate=Candidate(**old["candidate"])
    new=fit_profile(problems,candidate,metric,cpu_cap=job["cpu_remaining"],max_calls=job["calls_remaining"],
        starts=tuple(job["failed_fractions"]),max_iterations=contract["budgets"]["profile_max_iterations"])
    if new["fitted"]:new["selection_score"]=new["objective"]+new["parameter_count"]*math.log(metric.rows)
    choose_new=bool(new["fitted"] and new["selection_score"]<old["fit"]["selection_score"])
    chosen=dict(new if choose_new else old["fit"])
    chosen.update({"version":"WPHID_FIXED_METRIC_NUMERIC_BOUND_R1_ORIGINAL_BUDGET",
        "cpu_seconds":old["fit"]["cpu_seconds"]+new["cpu_seconds"],"wall_seconds":old["fit"]["wall_seconds"]+new["wall_seconds"],
        "calls":old["fit"]["calls"]+new["calls"],"restarts":old["fit"]["restarts"]+new["restarts"],
        "errors":old["fit"]["errors"]+new["errors"],"chosen_point_origin":"numeric_bound_r1" if choose_new else "original_saved_point",
        "original_and_continuation_cpu":{ "original":old["fit"]["cpu_seconds"],"remaining_cap":job["cpu_remaining"],"continuation":new["cpu_seconds"]},
        "source_error_history_retained":True})
    record={**old,"fit":chosen,"numerical_completion":{ "original_file":job["original_file"],"original_file_sha256":sha(ROOT/job["original_file"]),
        "repair_protocol_sha256":sha(REPAIR),"failed_fractions_only":job["failed_fractions"],"no_new_restart":True,"raw_continuation":new}}
    write(output/"fits"/job["filename"],record)
    return {"file":job["filename"],"new_point_selected":choose_new,"cpu_total":chosen["cpu_seconds"],"calls_total":chosen["calls"]}


def main():
    from reproducibility.coupled_solver_v1 import workflow
    repair=json.loads(REPAIR.read_text(encoding="utf-8"));oldp=json.loads((ROOT/repair["original_protocol"]).read_text(encoding="utf-8"))
    original=ROOT/repair["original_output"];output=ROOT/repair["output_root"];check_runtime(output);verify_freeze(original,oldp)
    if output.exists():raise SystemExit("completed/faulted R1 retained; no rerun")
    software=ROOT/"artifacts/paper_coupled_solver_v2/numeric_bound_software/qualification_20261013_attempt01"
    sr=json.loads((software/"execution-complete.json").read_text(encoding="utf-8"));sf=json.loads((software/"source-freeze.json").read_text(encoding="utf-8"))
    if not sr["software_passed"] or sha(software/"assessment.json")!=sr["assessment_sha256"] or any(sha(ROOT/i["path"])!=i["sha256"] for i in sf["sources"]):raise SystemExit("numeric repair not qualified")
    receipt=json.loads((original/"fits-complete.json").read_text(encoding="utf-8"))
    if any(sha(original/"fits"/i["path"])!=i["sha256"] for i in receipt["files"]):raise SystemExit("original completed fit changed")
    index=json.loads((original/"input-index.json").read_text(encoding="utf-8"));blocks={b["identity"]:b for b in index["blocks"]}
    output.mkdir(exist_ok=False);(output/"fits").mkdir()
    shutil.copytree(original/"public",output/"public")
    (output/"truth_scoring_only").mkdir()
    shutil.copyfile(original/"truth_scoring_only/truth.json",output/"truth_scoring_only/truth.json")
    shutil.copyfile(original/"input-index.json",output/"input-index.json")
    freeze=json.loads((original/"source-freeze.json").read_text(encoding="utf-8"))
    freeze["sources"]+= [{"path":p,"sha256":sha(ROOT/p)} for p in repair["sources"]]
    freeze.update({"original_freeze_sha256":sha(original/"source-freeze.json"),"numeric_repair_protocol_sha256":sha(REPAIR),"no_new_scientific_input":True})
    write(output/"source-freeze.json",freeze)
    jobs=[];copied=[]
    for item in receipt["files"]:
        name=item["path"];record=json.loads((original/"fits"/name).read_text(encoding="utf-8"));fit=record["fit"]
        fractions=[r["start_fraction"] for r in fit.get("restarts",[]) if r.get("status")=="domain_or_solver_failure" and r.get("error")==repair["required_error"]]
        remaining=oldp["budgets"]["cpu_seconds_per_program_candidate"]-fit.get("cpu_seconds",0.)
        calls=oldp["budgets"]["profile_max_calls"]-fit.get("calls",0)
        if record["program"]=="wphid" and fractions and remaining>=repair["minimum_remaining_cpu_to_start"] and calls>0:
            jobs.append({"output":str(output),"original_file":str((original/"fits"/name).relative_to(ROOT)),"filename":name,"block":blocks[record["block"]],
                "cpu_remaining":remaining,"calls_remaining":calls,"failed_fractions":fractions})
        else:
            shutil.copyfile(original/"fits"/name,output/"fits"/name);copied.append({"path":name,"original_sha256":item["sha256"],"no_rerun":True})
    write(output/"repair-plan-freeze.json",{"jobs":jobs,"original_records_reused":copied,"input_index_sha256":sha(output/"input-index.json"),
        "repair_protocol_sha256":sha(REPAIR),"unspent_cpu_total":sum(j["cpu_remaining"] for j in jobs),"all_original_budgets_unchanged":True})
    for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
    done=[]
    with ProcessPoolExecutor(max_workers=oldp["budgets"]["worker_processes"]) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:result=future.result()
            except Exception as error:
                shutil.copyfile(ROOT/job["original_file"],output/"fits"/job["filename"])
                result={"file":job["filename"],"correction_failed":str(error),"original_retained":True,"no_retry":True}
            done.append(result);print(json.dumps({"completed":len(done),"correction_jobs":len(jobs),**result}),flush=True)
    files=[{"path":p.name,"sha256":sha(p)} for p in sorted((output/"fits").glob("*.json"))]
    write(output/"fits-complete.json",{"status":"completed","files":files,"attempts":len(files),"source_freeze_sha256":sha(output/"source-freeze.json"),
        "original_fit_receipt_sha256":sha(original/"fits-complete.json"),"correction_jobs":len(jobs),"original_completed_records_reused":len(copied),
        "new_inputs_or_candidates":0,"corrections":done})
    p=dict(oldp);p["version"]="COUPLED_SOLVER_NUMERIC_BOUND_R1_COMPLETION_OF_FIXED_BATCH_20261013";p["output_root"]=repair["output_root"]
    workflow.assess(p,output);workflow.conditional_decision(p,output)


if __name__=="__main__":main()
