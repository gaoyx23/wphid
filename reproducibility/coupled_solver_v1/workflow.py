"""Exclusive, frozen prepare -> fit all -> sealed score -> adopt/confirm.

Worker functions read only public observations and fit contract. Clean states
and true parameters are never used for weights, starts or model selection.
Already-completed records are hash-checked and skipped, never refitted.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/"reproducibility/coupled_solver_v1/scientific-protocol-20261013.json"
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path)
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def check_runtime(output):
    import wphid
    if Path.cwd().resolve()!=ROOT or not Path(wphid.__file__).resolve().is_relative_to(ROOT/"src"):
        raise RuntimeError("unexpected cwd or main editable installation")
    if not Path(output).resolve().is_relative_to(ROOT/"artifacts/paper_coupled_solver_v1"):
        raise RuntimeError("output root must be independent and local")
    return {"cwd":str(ROOT),"wphid":str(Path(wphid.__file__).resolve()),"python":sys.executable,"output_root":str(Path(output).resolve())}


def dependencies():
    from reproducibility.coupled_observation_v1.qualify_software import verify_dependencies
    for p in ("reproducibility/coupled_declaration_v1/physical-candidates-protocol-20261013.json",
              "reproducibility/coupled_declaration_v1/censored-observation-protocol-20261013.json",
              "reproducibility/coupled_solver_v1/software-protocol-20261013.json"):
        spec=json.loads((ROOT/p).read_text(encoding="utf-8"));verify_dependencies(spec)
        folder=ROOT/spec["output_parent"]/("qualification_20261013_attempt01" if "coupled_solver" in p else "software_qualification_20261013_attempt01")
        receipt=json.loads((folder/"execution-complete.json").read_text(encoding="utf-8"))
        if not receipt["software_passed"] or sha(folder/"assessment.json")!=receipt["assessment_sha256"] or sha(folder/"source-freeze.json")!=receipt["source_freeze_sha256"]:
            raise RuntimeError("new software qualification missing or changed")
        frozen=json.loads((folder/"source-freeze.json").read_text(encoding="utf-8"))
        if any(sha(ROOT/item["path"])!=item["sha256"] for item in frozen["sources"]):
            raise RuntimeError("qualified source changed; no old test rerun")


def verify_freeze(output,protocol):
    frozen=json.loads((output/"source-freeze.json").read_text(encoding="utf-8"))
    if (sha(PROTOCOL)!=frozen["protocol_sha256"] or sha(ROOT/".gitattributes")!=frozen["gitattributes_sha256"]
        or any(sha(ROOT/i["path"])!=i["sha256"] for i in frozen["sources"])):
        raise RuntimeError("scientific freeze changed")
    return frozen


def prepare(protocol,output):
    import numpy as np
    from wphid.coupled_balance_v1.packet_geometry import declared_shell_mesh
    from reproducibility.coupled_solver_v1.generator import GEOMETRY,generate,observe
    dependencies()
    if output.exists():
        verify_freeze(output,protocol)
        receipt=json.loads((output/"prepare-complete.json").read_text(encoding="utf-8"))
        if sha(output/"input-index.json")!=receipt["input_index_sha256"]:raise RuntimeError("completed input index changed")
        index=json.loads((output/"input-index.json").read_text(encoding="utf-8"))
        if any(sha(output/i["path"])!=i["sha256"] for i in index["files"]):raise RuntimeError("completed input changed; no regeneration")
        print("completed hash-matched preparation reused",flush=True);return
    output.mkdir(parents=True,exist_ok=False)
    runtime=check_runtime(output)
    write(output/"source-freeze.json",{"protocol_sha256":sha(PROTOCOL),"sources":[{"path":p,"sha256":sha(ROOT/p)} for p in protocol["source_files"]],
        "runtime":runtime,"authorization":protocol["authorization"],"gitattributes_sha256":sha(ROOT/".gitattributes")})
    (output/"public").mkdir();(output/"truth_scoring_only").mkdir();(output/"fits").mkdir()
    write(output/"public/fit-contract.json",{k:protocol[k] for k in ("reference_metric_sigma","budgets","selection","shared_prior","candidate_space","estimation_criterion")})
    mesh,_=declared_shell_mesh(GEOMETRY,*protocol["data"]["grid"])
    times=np.linspace(0.,protocol["data"]["duration_seconds"],protocol["data"]["time_samples"])
    files=[];blocks=[];truth={};trajectory_count=0
    for case,role in enumerate(protocol["data"]["source_roles"]):
        source="both_shared" if role=="near_equivalent" else role
        clean=[];load=[];integration=[];parameters=None;coefficients=None
        for mission in range(protocol["data"]["mission_count"]):
            fields,loads,parameters,coefficients,receipt=generate(source,mesh,times,mission,near=role=="near_equivalent")
            clean.append(fields);load.append(loads);integration.append(receipt);trajectory_count+=1
        for repeat in range(protocol["data"]["noise_blocks_per_role"]):
            block=f"block_{case:02d}_{repeat:02d}";payload={"t":times};sensor=[]
            for mission,(fields,loads) in enumerate(zip(clean,load)):
                seed=protocol["data"]["noise_seed_root"]+case*100+repeat*10+mission
                observed,observed_loads,sigma,load_sigma,record=observe(fields,loads,np.random.default_rng(seed),protocol["observation_contract"])
                for key,value in observed.items():payload[f"m{mission}.state.{key}"]=value;payload[f"m{mission}.sigma.{key}"]=sigma[key]
                for key,value in observed_loads.items():payload[f"m{mission}.load.{key}"]=value;payload[f"m{mission}.load_sigma.{key}"]=load_sigma[key]
                sensor.append({**record,"seed":seed})
            path=f"public/{block}.npz"
            with (output/path).open("xb") as f:np.savez_compressed(f,**payload)
            files.append({"path":path,"sha256":sha(output/path)});blocks.append({"identity":block,"path":path,"sha256":sha(output/path),"mission_count":len(clean)})
            truth[block]={"role":role,"source":source,"parameters":parameters,"coefficients":coefficients,"integration":integration,"sensor":sensor}
        print(f"prepared scenario {case+1}/{len(protocol['data']['source_roles'])}; no fits or scores",flush=True)
    write(output/"truth_scoring_only/truth.json",truth)
    files.extend({"path":p,"sha256":sha(output/p)} for p in ("truth_scoring_only/truth.json","public/fit-contract.json"))
    write(output/"input-index.json",{"blocks":blocks,"files":files,"geometry":GEOMETRY,"weak_tests":{"window_samples":protocol["data"]["window_samples"],"stride_samples":protocol["data"]["stride_samples"]},
        "trajectories":trajectory_count,"noise_blocks":len(blocks),"scientific_identity":"new controlled development, not independent engineering evidence"})
    write(output/"prepare-complete.json",{"status":"completed","input_index_sha256":sha(output/"input-index.json"),"source_freeze_sha256":sha(output/"source-freeze.json"),
        "new_trajectories":trajectory_count,"new_noise_blocks":len(blocks),"new_fits":0})


def problems_from_public(output,block,index):
    import numpy as np
    from wphid.coupled_balance_v1 import WeakProjector,temporal_tests
    from wphid.coupled_balance_v1.assembler import BoundaryInputs,PRESSURES
    from wphid.coupled_balance_v1.operators import SIDES,Modulation
    from wphid.coupled_balance_v1.packet_geometry import declared_shell_mesh
    from wphid.coupled_solver_v1 import ValueProblem
    with np.load(output/block["path"],allow_pickle=False) as data:
        shape=data["m0.state.T"].shape;nt,ns,nx=shape
        mesh,_=declared_shell_mesh(index["geometry"],ns,nx)
        tests=temporal_tests(data["t"],**index["weak_tests"])
        projector=WeakProjector(mesh,tests,np.eye(ns*nx).reshape(ns*nx,ns,nx))
        problems=[];sigmas=[];load_sigmas=[]
        for mission in range(block["mission_count"]):
            fields={key:data[f"m{mission}.state.{key}"].copy() for key in ("carbon","pO2","pH2O","pProducts","T","crack","stress")}
            loads={key:data[f"m{mission}.load.{key}"].copy() for key in ("internal_oxygen_pressure","external_steam_pressure","external_oxygen_pressure","surface_heat_flux")}
            external={("pO2","inner"):loads["internal_oxygen_pressure"],("pH2O","inner"):loads["external_steam_pressure"],("pProducts","inner"):np.zeros((nt,ns)),("pO2","outer"):loads["external_oxygen_pressure"]}
            boundary=BoundaryInputs(external,{k:Modulation(np.ones((nt,ns))) for k in external},frozenset((f,s) for f in PRESSURES for s in SIDES if (f,s) not in external),
                {"inner":loads["surface_heat_flux"]},{"outer":np.full((nt,ns),300.)},frozenset(("inlet","outlet")))
            p=ValueProblem(projector,fields,boundary);p.raw_loads=loads;problems.append(p)
            sigmas.append({key:data[f"m{mission}.sigma.{key}"].copy() for key in fields})
            load_sigmas.append({key:data[f"m{mission}.load_sigma.{key}"].copy() for key in loads})
    return problems,sigmas,load_sigmas


def worker(job):
    output=Path(job["output"]);block=job["block"]
    from wphid.coupled_solver_v1 import Candidate,common_metric,fit_profile,fit_adam
    check_runtime(output)
    contract=json.loads((output/"public/fit-contract.json").read_text(encoding="utf-8"))
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"))
    if sha(output/block["path"])!=block["sha256"]:raise RuntimeError("input changed before fit")
    problems,_,_=problems_from_public(output,block,index)
    metric=common_metric(problems,contract["reference_metric_sigma"])
    candidate=Candidate(**job["candidate"]);budget=contract["budgets"]
    try:
        if job["program"]=="wphid":
            fit=fit_profile(problems,candidate,metric,cpu_cap=budget["cpu_seconds_per_program_candidate"],max_calls=budget["profile_max_calls"],max_iterations=budget["profile_max_iterations"],starts=tuple(budget["initial_fractions"]))
        else:
            fit=fit_adam(problems,candidate,metric,cpu_cap=budget["cpu_seconds_per_program_candidate"],max_calls=budget["adam_max_calls"],starts=tuple(budget["initial_fractions"]),learning_rate=budget["adam_learning_rate"])
        if fit["fitted"]:fit["selection_score"]=fit["objective"]+fit["parameter_count"]*__import__("math").log(metric.rows)
    except Exception as error:
        fit={"fitted":False,"error":str(error),"traceback":traceback.format_exc(),"retries":0}
    record={"block":block["identity"],"input_sha256":block["sha256"],"program":job["program"],"candidate":job["candidate"],
        "candidate_identity":candidate.identity,"fit":fit,"runtime":check_runtime(output),"truth_used_for_fit_or_selection":False,"protocol_sha256":sha(PROTOCOL)}
    write(output/"fits"/job["filename"],record)
    return {"filename":job["filename"],"fitted":fit["fitted"],"cpu_seconds":fit.get("cpu_seconds",0.)}


def run_fits(protocol,output):
    from dataclasses import asdict
    from wphid.coupled_solver_v1 import candidate_space
    verify_freeze(output,protocol)
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"))
    jobs=[];existing=[]
    for block in index["blocks"]:
        if sha(output/block["path"])!=block["sha256"]:raise RuntimeError("fixed observation changed")
        for cindex,candidate in enumerate(candidate_space()):
            for program in ("wphid","adam"):
                filename=f"{block['identity']}_c{cindex:02d}_{program}.json"
                if (output/"fits"/filename).exists():
                    record=json.loads((output/"fits"/filename).read_text(encoding="utf-8"))
                    if record["input_sha256"]!=block["sha256"] or record["protocol_sha256"]!=sha(PROTOCOL):raise RuntimeError("saved completed fit identity changed")
                    existing.append(filename);continue
                jobs.append({"output":str(output),"block":block,"candidate":asdict(candidate),"program":program,"filename":filename})
    if (output/"fits-complete.json").exists():
        receipt=json.loads((output/"fits-complete.json").read_text(encoding="utf-8"))
        if jobs or any(sha(output/"fits"/i["path"])!=i["sha256"] for i in receipt["files"]):raise RuntimeError("completed fit set changed")
        print("all completed fit hashes matched, none rerun",flush=True);return
    done=len(existing);total=done+len(jobs)
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
    with ProcessPoolExecutor(max_workers=protocol["budgets"]["worker_processes"]) as executor:
        futures={executor.submit(worker,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:result=future.result()
            except Exception as error:
                # A crashed job is retained, not automatically resubmitted.
                result={"filename":job["filename"],"fitted":False,"error":str(error)}
                if not (output/"fits"/job["filename"]).exists():
                    write(output/"fits"/job["filename"],{"block":job["block"]["identity"],"input_sha256":job["block"]["sha256"],"program":job["program"],"candidate":job["candidate"],"fit":result,"protocol_sha256":sha(PROTOCOL),"worker_crashed_no_retry":True})
            done+=1
            print(json.dumps({"completed":done,"total":total,**result},ensure_ascii=False),flush=True)
    files=[{"path":p.name,"sha256":sha(p)} for p in sorted((output/"fits").glob("*.json"))]
    write(output/"fits-complete.json",{"status":"completed","files":files,"attempts":total,"source_freeze_sha256":sha(output/"source-freeze.json"),"truth_scoring_opened_during_fits":False})


def errors_for_truth(candidate,fit,truth,precision):
    from wphid.coupled_solver_v1.model import parameter_box
    if not fit.get("fitted"):return {"assessable":False,"precision_pass":False,"reason":"fit failed"}
    role=truth["role"];expected=truth["source"]
    correct=candidate.source==expected and candidate.gas=="combined_opening" and candidate.thermal=="stiffness" and candidate.flux=="Darcy"
    errors={};critical=[];nuisance={}
    for key,value in fit["parameters"].items():
        if key not in truth["parameters"]:continue
        target=truth["parameters"][key];relative=abs(value-target)/abs(target) if target else abs(value-target)
        if key.startswith("kinetic."):
            threshold=precision["activation_energy_relative" if key.endswith("energy_1e5") else ("pressure_order_relative" if key.endswith("pressure_order") else "carbon_order_relative")]
            errors[key]=relative;critical.append(relative/threshold)
        elif key=="reaction.relative_rate":errors[key]=relative;critical.append(relative/precision["relative_rate_relative"])
        else:nuisance[key]=relative
    for key,value in fit["coefficients"].items():
        if key not in truth["coefficients"]:continue
        target=truth["coefficients"][key];relative=abs(value-target)/abs(target) if target else abs(value-target)
        if key in ("gas_scale","surface_transfer","thermal_scale","heat_input_scale","radiation_scale","heat_capacity_contrast"):
            nuisance[key]=relative
        else:errors[key]=relative;critical.append(relative/precision["active_source_response_relative"])
    # Free source representation can be compared with the same true-shared
    # physical responses without choosing a truth-informed restart.
    if expected=="both_shared" and candidate.source=="both_free":
        r=truth["parameters"]["reaction.relative_rate"];a=truth["coefficients"]
        targets={"carbon.oxygen_response":a["carbon_rate_scale"],"carbon.steam_response":a["carbon_rate_scale"]*r,
            "pO2.oxygen_response":a["oxygen_response"],"pH2O.steam_response":a["steam_response"],
            "pProducts.oxygen_response":a["product_response"],"pProducts.steam_response":a["product_response"]*r,
            "T.oxygen_response":a["heat_response"],"T.steam_response":a["heat_response"]*r}
        for k,target in targets.items():
            relative=abs(fit["coefficients"][k]-target)/abs(target);errors[k]=relative;critical.append(relative/precision["active_source_response_relative"])
    score=max(critical) if critical else None
    return {"assessable":score is not None,"correct_complete_candidate":correct,"reaction_score":score,
        "precision_pass":bool(score is not None and score<=1.),"joint_structure_and_precision":bool(correct and score is not None and score<=1.),
        "reaction_relative_errors":errors,"nuisance_relative_errors":nuisance,"role":role,
        "all_unknown_properties_claimed_precisely_recovered":False}


def assess(protocol,output):
    import numpy as np
    from wphid.coupled_solver_v1 import Candidate,common_metric,information_at_fit
    from wphid.coupled_solver_v1.solver import stacked
    from wphid.coupled_solver_v1.uncertainty import raw_at_fit
    from wphid.coupled_likelihood_v1 import evaluate_quasi_likelihood
    verify_freeze(output,protocol)
    if (output/"assessment.json").exists():
        receipt=json.loads((output/"execution-complete.json").read_text(encoding="utf-8"))
        if sha(output/"assessment.json")!=receipt["assessment_sha256"]:raise RuntimeError("assessment changed")
        print("completed assessment hash matched, no rescoring",flush=True);return
    receipt=json.loads((output/"fits-complete.json").read_text(encoding="utf-8"))
    if any(sha(output/"fits"/i["path"])!=i["sha256"] for i in receipt["files"]):raise RuntimeError("sealed fit changed before scoring")
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"))
    truth=json.loads((output/"truth_scoring_only/truth.json").read_text(encoding="utf-8"))
    rows=[];pairs=[];uncertainty=[];replay=[]
    for block in index["blocks"]:
        problems,sigmas,load_sigmas=problems_from_public(output,block,index);metric=common_metric(problems,protocol["reference_metric_sigma"])
        selected={};allrecords={}
        for program in ("wphid","adam"):
            records=[json.loads(p.read_text(encoding="utf-8")) for p in (output/"fits").glob(block["identity"]+"_*_"+program+".json")]
            fitted=sorted((r for r in records if r["fit"]["fitted"]),key=lambda r:(r["fit"]["selection_score"],r.get("candidate_identity","")))
            allrecords[program]=records
            if not fitted:
                rows.append({"block":block["identity"],"program":program,"role":truth[block["identity"]]["role"],"selected":None,"precision_pass":False,"joint_structure_and_precision":False});continue
            chosen=fitted[0];fit=chosen["fit"];candidate=Candidate(**chosen["candidate"]);selected[program]=chosen
            b,G,_,keys=stacked(problems,candidate,fit["parameters"])
            a=np.array([fit["coefficients"][k] for k in keys]);objective=float(np.sum(metric.apply(b-G@a)**2))
            replay_defect=abs(objective-fit["objective"])/max(abs(objective),1.)
            replay.append({"block":block["identity"],"program":program,"relative_objective_defect":replay_defect,"passed":replay_defect<=1e-9})
            grade=errors_for_truth(candidate,fit,truth[block["identity"]],protocol["precision"])
            competitors=[r["candidate"] for r in fitted if r["fit"]["selection_score"]-fit["selection_score"]<=protocol["selection"]["competition_score_gap"]]
            info=information_at_fit(problems,candidate,metric,fit)
            rows.append({"block":block["identity"],"program":program,"role":truth[block["identity"]]["role"],"selected":chosen["candidate"],
                "score":fit["selection_score"],"competitors":competitors,"local_information":info,**grade})
            if program=="wphid":
                for mission,p in enumerate(problems):
                    try:
                        model,lin,F,domain=raw_at_fit(p,candidate,fit["parameters"],sigmas[mission],load_sigmas[mission])
                        aa=np.array([fit["coefficients"][k] for k in model.column_keys])
                        q=evaluate_quasi_likelihood(model,lin,aa,F,max_covariance_pairs=20_000_000)
                        uncertainty.append({"block":block["identity"],"mission":mission,"qualified_computation":True,"objective":q.objective,
                            "covariance_nnz":q.covariance_nnz,"domain":domain,"used_for_fit_or_selection":False,
                            "sensor_variance_is_observation_plugin":True,"nonlinear_mean_bias_corrected":False})
                    except Exception as error:
                        uncertainty.append({"block":block["identity"],"mission":mission,"qualified_computation":False,"error":str(error),
                            "used_for_fit_or_selection":False,"jitter_or_row_deletion":False})
        if truth[block["identity"]]["role"]=="both_shared":
            reference={r["candidate"]["source"]:r for r in allrecords["wphid"] if r["candidate"]["gas"]=="combined_opening" and r["candidate"]["thermal"]=="stiffness" and r["candidate"]["flux"]=="Darcy"}
            left=errors_for_truth(Candidate("both_shared"),reference["both_shared"]["fit"],truth[block["identity"]],protocol["precision"])
            right=errors_for_truth(Candidate("both_free"),reference["both_free"]["fit"],truth[block["identity"]],protocol["precision"])
            win=bool(left["assessable"] and right["assessable"] and right["reaction_score"]>0 and left["reaction_score"]<=.8*right["reaction_score"] and left["precision_pass"])
            pairs.append({"block":block["identity"],"shared":left,"free":right,"predeclared_component_win":win})
    positive_roles=("both_shared","both_free","both_independent","oxygen_only","steam_only","both_oxygen_heat")
    counts={program:{"positive_total":12,"positive_joint_pass":sum(r.get("joint_structure_and_precision",False) for r in rows if r["program"]==program and r["role"] in positive_roles),
        "positive_parameter_pass":sum(r.get("precision_pass",False) for r in rows if r["program"]==program and r["role"] in positive_roles),
        "null_correct":sum(bool(r["selected"] and r["selected"]["source"]=="null") for r in rows if r["program"]==program and r["role"]=="null"),
        "near_competition":sum({"both_shared","both_swapped_product"}.issubset({c["source"] for c in r.get("competitors",[]) if c["gas"]=="combined_opening" and c["thermal"]=="stiffness" and c["flux"]=="Darcy"}) for r in rows if r["program"]==program and r["role"]=="near_equivalent")} for program in ("wphid","adam")}
    ratios=[];all_pair_assessable=True
    for block in index["blocks"]:
        if truth[block["identity"]]["role"] not in positive_roles:continue
        paired={r["program"]:r for r in rows if r["block"]==block["identity"]}
        left=paired["wphid"].get("reaction_score");right=paired["adam"].get("reaction_score")
        if left is None or right is None or right<=0:all_pair_assessable=False
        else:ratios.append(left/right)
    median_ratio=float(np.median(ratios)) if ratios else None
    gates={"positive_joint_precision":counts["wphid"]["positive_joint_pass"]>=10,"null":counts["wphid"]["null_correct"]==2,
        "near_equivalent":counts["wphid"]["near_competition"]==2,"source_component":len(pairs)==2 and all(p["predeclared_component_win"] for p in pairs),
        "matched_external":bool(all_pair_assessable and median_ratio is not None and median_ratio<=1.1 and counts["wphid"]["positive_parameter_pass"]>=counts["adam"]["positive_parameter_pass"]),
        "replay_and_source_identity":all(r["passed"] for r in replay)}
    assessment={"protocol":protocol["version"],"status":"completed","counts":counts,"source_component_pairs":pairs,"selected_records":rows,
        "matched_selected_reaction_score_median_ratio":median_ratio,"all_external_positive_pairs_assessable":all_pair_assessable,
        "adoption_gates":gates,"adopted":all(gates.values()),"raw_covariance_diagnostics":uncertainty,"selected_fit_replays":replay,
        "scientific_fit_attempts":receipt["attempts"],"finite_fit_count":sum(json.loads((output/"fits"/i["path"]).read_text(encoding="utf-8"))["fit"]["fitted"] for i in receipt["files"]),
        "new_trajectories":index["trajectories"],"new_noise_blocks":index["noise_blocks"],"original_eight_package_count":"6/8",
        "scope":protocol["scope"],"native_ADAM_or_exact_MLE_advantage_established":False,"declared_fixed_metric_approximation":protocol["estimation_criterion"],
        "no_post_result_extension_or_tuning":True}
    write(output/"assessment.json",assessment)
    write(output/"execution-complete.json",{"status":"completed","assessment_sha256":sha(output/"assessment.json"),"source_freeze_sha256":sha(output/"source-freeze.json"),
        "fits_complete_sha256":sha(output/"fits-complete.json"),"adopted":assessment["adopted"]})
    print(json.dumps({k:assessment[k] for k in ("counts","adoption_gates","adopted","scientific_fit_attempts","finite_fit_count","new_trajectories","new_noise_blocks")},ensure_ascii=False,indent=2),flush=True)


def conditional_decision(protocol,output):
    assessment=json.loads((output/"assessment.json").read_text(encoding="utf-8"))
    path=output/"confirmation-decision.json"
    if path.exists():return
    if not assessment["adopted"]:
        write(path,{"development_status":"completed_not_adopted","independent_confirmation":"not_run_dependency_gate_failed",
            "engineering_confirmation":"not_run_dependency_gate_failed","reason":"fixed development adoption gate failed; no batch extension or input regeneration",
            "failed_gates":[k for k,v in assessment["adoption_gates"].items() if not v],"new_confirmation_fits":0,"main_directory_adoption":False})
    else:
        # Confirmation is a distinct predeclared phase; never silently inherit
        # old qualification. The caller continues it only after this receipt.
        write(path,{"development_status":"adopted_in_worktree_only","independent_confirmation":"required_predeclared_new_660s_inputs",
            "engineering_confirmation":"required_already_viewed_R100_compatibility_challenge_not_independent","main_directory_adoption":False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("prepare","fit","assess","decision","all"),required=True);args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"));output=ROOT/protocol["output_root"];check_runtime(output)
    for stage,function in (("prepare",prepare),("fit",run_fits),("assess",assess),("decision",conditional_decision)):
        if args.stage in (stage,"all"):function(protocol,output)


if __name__=="__main__":main()
