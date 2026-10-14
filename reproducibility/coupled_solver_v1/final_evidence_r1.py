"""Final same-physical-endpoint and budget/provenance coverage, no fits."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]


def main():
    from reproducibility.coupled_solver_v1.workflow import sha,write,verify_freeze
    from reproducibility.coupled_solver_v1.physical_endpoints import grade
    repair=json.loads((ROOT/"reproducibility/coupled_solver_v1/numeric-bound-completion-r1-protocol-20261013.json").read_text(encoding="utf-8"))
    protocol=json.loads((ROOT/repair["original_protocol"]).read_text(encoding="utf-8"))
    output=ROOT/repair["output_root"];original=ROOT/repair["original_output"];verify_freeze(output,protocol)
    if (output/"final-evidence.json").exists():raise SystemExit("completed final evidence retained, no rescoring")
    receipt=json.loads((output/"execution-complete.json").read_text(encoding="utf-8"))
    if sha(output/"assessment.json")!=receipt["assessment_sha256"]:raise SystemExit("assessment changed")
    fitreceipt=json.loads((output/"fits-complete.json").read_text(encoding="utf-8"))
    if any(sha(output/"fits"/i["path"])!=i["sha256"] for i in fitreceipt["files"]):raise SystemExit("fit identity changed")
    assessment=json.loads((output/"assessment.json").read_text(encoding="utf-8"));truth=json.loads((output/"truth_scoring_only/truth.json").read_text(encoding="utf-8"))
    index=json.loads((output/"input-index.json").read_text(encoding="utf-8"))
    if any(sha(output/i["path"])!=i["sha256"] for i in index["files"]):raise SystemExit("same fixed inputs changed")
    records={}
    for i in fitreceipt["files"]:
        r=json.loads((output/"fits"/i["path"]).read_text(encoding="utf-8"));records[(r["block"],r["program"],tuple(r["candidate"].values()))]=r
    selected=[]
    for row in assessment["selected_records"]:
        if row["selected"] is None:continue
        r=records[(row["block"],row["program"],tuple(row["selected"].values()))]
        result=grade(row["selected"]["source"],r["fit"],truth[row["block"]],protocol["precision"])
        selected.append({"block":row["block"],"program":row["program"],"role":row["role"],"candidate":row["selected"],
            "source_identity_correct":row["selected"]["source"]==truth[row["block"]]["source"],**result})
    sourcepairs=[]
    for block in (b for b,t in truth.items() if t["role"]=="both_shared"):
        pair={"block":block}
        for source in ("both_shared","both_free"):
            r=records[(block,"wphid",(source,"combined_opening","stiffness","Darcy"))]
            pair[source]=grade(source,r["fit"],truth[block],protocol["precision"])
        sourcepairs.append(pair)
    stats={};budget_ok=True
    for program in ("wphid","adam"):
        fits=[r["fit"] for r in records.values() if r["program"]==program]
        cpu=[f["cpu_seconds"] for f in fits]
        for f in fits:
            if program=="wphid" and f["calls"]>protocol["budgets"]["profile_max_calls"]:budget_ok=False
            if "original_and_continuation_cpu" in f:
                detail=f["original_and_continuation_cpu"]
                if abs(detail["remaining_cap"]-(15.-detail["original"]))>1e-9:budget_ok=False
        stats[program]={"candidate_program_records":len(fits),"cpu_sum_seconds":sum(cpu),"cpu_median_seconds":float(np.median(cpu)),
            "cpu_maximum_seconds":max(cpu),"budget_truncated_records":sum(any(s.get("budget_exhausted",False) or s.get("status")=="budget_exhausted" for s in f["restarts"]) for f in fits),
            "retained_original_restart_errors":sum(bool(f.get("errors")) for f in fits),"full_program_preprocessing_CPU_included":False,
            "call_cap_satisfied":program!="wphid" or max(f["calls"] for f in fits)<=protocol["budgets"]["profile_max_calls"]}
    positives={p:sum(r["complete_physical_parameter_pass"] for r in selected if r["program"]==p and r["role"] not in ("null","near_equivalent")) for p in ("wphid","adam")}
    sourcecorrect={p:sum(r["source_identity_correct"] for r in selected if r["program"]==p and r["role"] not in ("null","near_equivalent")) for p in ("wphid","adam")}
    closeout=json.loads((original/"closeout-audit.json").read_text(encoding="utf-8"))
    result={"status":"completed","full_physical_parameter_positive_counts":positives,"positive_denominator":12,"positive_source_identity_counts":sourcecorrect,
        "selected_records":selected,"same_observation_source_component_pairs":sourcepairs,"program_statistics":stats,
        "original_batch":{ "physical_trajectories":24,"noise_blocks":16,"candidate_program_combinations":384},
        "arithmetic_repair":{ "interrupted_candidate_programs_completed":fitreceipt["correction_jobs"],"original_records_reused_without_refit":fitreceipt["original_completed_records_reused"],
            "new_inputs":0,"new_trajectories":0,"new_noise_blocks":0,"new_candidates":0,"new_start_fractions":0,
            "remaining_original_CPU_and_calls_only":budget_ok,"better_training_points_selected":sum(r.get("new_point_selected",False) for r in fitreceipt["corrections"])},
        "adoption_gates_unchanged":assessment["adoption_gates"],"adopted":assessment["adopted"],
        "old_score_median_ratio_not_a_complete_physical_endpoint_advantage":True,
        "scientific_readiness":{ "useful_precision_equivalence_to_native_ADAM_established":False,"independent_confirmation":False,"engineering_confirmation":False,
            "fixed_reference_metric_is_actual_residual_covariance":False},
        "clean_compatibility_reused_without_repeating_24_checks":closeout["saved_clean_truth_point_compatibility"],
        "original_eight_package_count":"6/8","old_384_source_results_retained":True,"source_sha256":sha(Path(__file__))}
    write(output/"final-evidence.json",result)
    with (output/"selected-physical-parameters.csv").open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=("block","role","program","source","gas","thermal","flux","physical_score","complete_physical_parameter_pass","missing_count","zero_path_violation_count"));w.writeheader()
        for r in selected:w.writerow({"block":r["block"],"role":r["role"],"program":r["program"],**r["candidate"],"physical_score":r.get("reaction_score"),
            "complete_physical_parameter_pass":r["complete_physical_parameter_pass"],"missing_count":len(r.get("missing_coordinates",[])),"zero_path_violation_count":len(r.get("zero_path_violations",[]))})
    write(output/"final-evidence-complete.json",{"status":"completed","final_evidence_sha256":sha(output/"final-evidence.json"),"csv_sha256":sha(output/"selected-physical-parameters.csv"),"new_fits":0})
    print(json.dumps({"physical_parameter_pass":positives,"positive_source_identity":sourcecorrect,"statistics":stats,"budget_coverage_valid":budget_ok,"adopted":assessment["adopted"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
