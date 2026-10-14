"""Assess frozen completed estimates; reference truth never enters optimizers."""

import csv
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(Path(__file__).parent))
sys.path.insert(2, str(ROOT))

import numpy as np
from scipy.stats import binomtest

from scientific_io import load_conditions, save_json, sha
from run_confirmation import PROTOCOL, OUTPUT, read_json, verify_freeze, jobs_for, output_path
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import Candidate, Mechanism, FittedCandidate
from wphid.final_candidate_v1.pipeline import prepare_views, training_diagnostics


def assess_fit(record, role):
    # This reference is imported only after immutable estimates exist.
    from experiments import paper_h3_joint_family_topology_partition as truth
    fit = record.get("fit")
    result = {"role": role, "success": record["status"] == "success", "assessable": False,
              "precision_pass": False, "case_score": None,
              "parameter_errors_by_equation": None, "amplitude_errors": None,
              "base_coefficient_errors": None}
    if fit is None:
        return result
    mechanisms = fit["hypothesis"]
    if role == "null":
        result["null_selected"] = not mechanisms
        return result
    if (len(mechanisms) != 1 or mechanisms[0]["family"] != truth.TRUTH_FAMILY
            or any(g is None for g in mechanisms[0]["groups"])):
        result["incompatible_decomposition"] = True
        return result
    mechanism = mechanisms[0]
    parameters = np.asarray([mechanism["internal_parameters_by_group"][g] for g in mechanism["groups"]])
    coefficients = fit["physical_coefficients"]
    amplitudes = np.asarray([c[-1] for c in coefficients])
    true_parameters = np.asarray(truth.TRUTH_CASES[role]["parameters"])
    true_amplitudes = np.asarray(truth.TRUTH_AMPLITUDES)
    if (parameters.shape != (3,3) or not np.isfinite(parameters).all() or not np.isfinite(amplitudes).all()
            or np.any(np.abs(amplitudes) <= 1e-12)):
        result["invalid_or_zero_amplitude_decomposition"] = True
        return result
    errors = np.abs(parameters-true_parameters)/np.maximum(np.abs(true_parameters),1e-12)
    amp_errors = np.abs(amplitudes-true_amplitudes)/np.abs(true_amplitudes)
    gates = PROTOCOL["precision_gates"]
    limits = np.asarray([gates["theta"], gates["oxygen_order"], gates["damage_order"]])
    score = max(float(np.max(errors/limits)), float(np.max(amp_errors/gates["active_reaction_amplitude"])))
    base_errors = []
    for field, c in zip(truth.FIELDS, coefficients):
        expected = np.asarray(truth.h1.CLOSED["truth_coefficients"][field][:-1])
        base_errors.append((np.abs(np.asarray(c[:-1])-expected)/np.maximum(np.abs(expected),1e-12)).tolist())
    result.update({"assessable": True, "precision_pass": score <= 1., "case_score": score,
                   "parameter_errors_by_equation": errors.tolist(), "amplitude_errors": amp_errors.tolist(),
                   "base_coefficient_errors": base_errors,
                   "maximum_parameter_relative_error": float(np.max(errors)),
                   "maximum_amplitude_relative_error": float(np.max(amp_errors)),
                   "exact_family_topology_partition": mechanism["groups"] == list(truth.TRUTH_CASES[role]["partition"]),
                   "optimizer_status": fit["optimizer_status"],
                   "cpu_seconds": fit["cpu_seconds"],
                   "budget_exhausted": fit["optimizer_status"] == "budget_exhausted",
                   "frozen_heldout_residual": record["frozen_heldout"]["equation_weak_target_relative_residual"]})
    return result


def restore_fit(record):
    families = {f.name:f for f in paper_families()}
    data = record["fit"]
    candidate = Candidate(tuple(Mechanism(m["mechanism"], families[m["family"]], tuple(m["groups"]))
                                for m in data["hypothesis"]))
    return FittedCandidate(candidate, np.asarray(record["flat_parameters"]),
                           tuple(np.asarray(c) for c in data["scaled_coefficients"]),
                           tuple(data["equation_training_rss"]), data["training_objective"],
                           data["common_final_score"], data["nominal_parameter_count"],
                           data["training_scale_sha256"], data["optimizer_status"], data["objective_calls"],
                           data["cpu_seconds"], data["wall_seconds"], tuple(data["restarts"]))


def descriptive(rows):
    valid = [r for r in rows if r["assessable"]]
    return {"n": len(rows), "assessable": len(valid), "precision_passes": sum(r["precision_pass"] for r in rows),
            "failed_or_invalid": len(rows)-len(valid),
            "budget_exhausted": sum(r.get("budget_exhausted",False) for r in rows),
            "median_case_score": float(np.median([r["case_score"] for r in valid])) if valid else None,
            "worst_case_score": max((r["case_score"] for r in valid), default=None),
            "median_maximum_parameter_relative_error": float(np.median([r["maximum_parameter_relative_error"] for r in valid])) if valid else None,
            "median_maximum_amplitude_relative_error": float(np.median([r["maximum_amplitude_relative_error"] for r in valid])) if valid else None,
            "median_cpu_seconds": float(np.median([r["cpu_seconds"] for r in valid])) if valid else None}


def main():
    inputs = verify_freeze()
    completed = read_json(OUTPUT / "execution-complete.json")
    destination = OUTPUT / "assessment.json"
    if destination.exists():
        raise FileExistsError("Assessment already saved; read the immutable result")
    if (completed["source_freeze_sha256"] != sha(OUTPUT / "source-freeze.json")
            or completed["input_manifest_sha256"] != sha(OUTPUT / "input-manifest.json")):
        raise ValueError("Execution identities differ from current freeze")
    records, identities = {}, {}
    for stage in ("Q1","Q2","Q3"):
        for job in jobs_for(stage, inputs):
            path = output_path(job)
            record = read_json(path)
            if (record["job"] != job or record["input_sha256"] != job["input"]["sha256"]
                    or record["source_freeze_sha256"] != sha(OUTPUT / "source-freeze.json")
                    or not record["completed"]):
                raise ValueError("Unbound, missing or incomplete scientific estimate")
            records[job["id"]] = record
            identities[path.relative_to(ROOT).as_posix()] = sha(path)
    q1_rows, lookup = [], {}
    for seed in PROTOCOL["Q1"]["block_seeds"]:
        for role in PROTOCOL["Q1"]["roles"]:
            for program in ("wphid","adam"):
                record = records[str(seed)+"_"+role+"_"+program]
                row = {"seed": seed, "program": program, **assess_fit(record,role)}
                q1_rows.append(row)
                lookup[seed,role,program] = row
    blocks = []
    for seed in PROTOCOL["Q1"]["block_seeds"]:
        scores, valid = {}, True
        for program in ("wphid","adam"):
            rows = [lookup[seed,role,program] for role in PROTOCOL["Q1"]["roles"]]
            assessable = all(r["assessable"] for r in rows)
            valid &= assessable
            scores[program] = max(r["case_score"] for r in rows) if assessable else None
        win = bool(valid and scores["adam"] > 0 and scores["wphid"] <= .9*scores["adam"])
        blocks.append({"seed": seed,"scores":scores,"both_assessable":valid,"wphid_win":win})
    wins = sum(b["wphid_win"] for b in blocks)
    role_summary = {role:{program:descriptive([r for r in q1_rows if r["role"] == role and r["program"] == program])
                          for program in ("wphid","adam")} for role in PROTOCOL["Q1"]["roles"]}
    rate_gate = all(v["wphid"]["precision_passes"] >= v["adam"]["precision_passes"] for v in role_summary.values())
    q1 = {"rows":q1_rows,"blocks":blocks,"role_summary":role_summary,
          "total_summary":{p:descriptive([r for r in q1_rows if r["program"] == p]) for p in ("wphid","adam")},
          "block_wins":wins,"block_denominator":18,
          "one_sided_binomial_p":float(binomtest(wins,18,.5,alternative="greater").pvalue),
          "role_pass_rate_gate":rate_gate,"declared_conditional_advantage_gate_pass":wins>=13 and rate_gate,
          "native_end_to_end_ADAM_comparison":False,"equal_maximum_budget_not_equal_convergence":True}
    q2_rows, selection = [], []
    for seed in PROTOCOL["Q2"]["reuse_Q1_block_seeds"]:
        for role in PROTOCOL["Q2"]["roles"]:
            alternatives = []
            for partition in PROTOCOL["Q2"]["partitions"]:
                key = str(seed)+"_"+role+("_wphid" if partition == role else "_as_"+partition)
                record = records[key]
                row = {"seed":seed,"assumed_partition":partition,"Q1_fit_reused":partition==role,
                       "result_source":output_path(record["job"]).relative_to(ROOT).as_posix(),**assess_fit(record,role)}
                q2_rows.append(row)
                if record["status"] == "success":
                    alternatives.append((record["fit"]["common_final_score"],partition))
            alternatives.sort()
            selection.append({"seed":seed,"role":role,"selected_partition":alternatives[0][1] if alternatives else None,
                              "correct":bool(alternatives and alternatives[0][1]==role),
                              "all_three_fits_successful":len(alternatives)==3})
    q2_summary = {role:{partition:descriptive([r for r in q2_rows if r["role"]==role and r["assumed_partition"]==partition])
                       for partition in PROTOCOL["Q2"]["partitions"]} for role in PROTOCOL["Q2"]["roles"]}
    shared_pairs = []
    for seed in PROTOCOL["Q2"]["reuse_Q1_block_seeds"]:
        same = {r["assumed_partition"]:r for r in q2_rows if r["seed"]==seed and r["role"]=="shared"}
        a,b = same["shared"],same["independent"]
        shared_pairs.append({"seed":seed,"correct_shared_case_score":a["case_score"],"independent_case_score":b["case_score"],
                             "relative_case_score_reduction":1-a["case_score"]/b["case_score"] if a["assessable"] and b["assessable"] and b["case_score"]>0 else None})
    q2 = {"rows":q2_rows,"summary":q2_summary,"training_score_partition_selection":selection,
          "selected_correct":sum(r["correct"] for r in selection),"selection_denominator":len(selection),
          "shared_vs_saturated_pairs":shared_pairs,"new_fits":36,"Q1_reused_fits":18,
          "single_factor_only":True,"formal_significance_claim":False}
    q3_rows = []
    q3_started = time.process_time()
    for input_record in [r for r in inputs["records"] if r["stage"]=="Q3"]:
        seed,role = input_record["seed"],input_record["role"]
        estimates = [records[str(seed)+"_"+role+"_candidate_"+str(i).zfill(2)] for i in range(43)]
        successful = [r for r in estimates if r["status"]=="success"]
        row = {"seed":seed,"role":role,"candidates_declared":43,"candidates_completed":len(estimates),
               "successful_candidates":len(successful),"complete_search":len(successful)==43,
               "all_estimates_CPU_seconds":sum(r["total_worker_job_cpu_seconds"] for r in estimates),
               "budget_exhausted_candidates":sum(r.get("fit",{}).get("optimizer_status")=="budget_exhausted" for r in estimates)}
        if successful:
            ordered = sorted(successful,key=lambda r:(r["fit"]["common_final_score"],str(restore_fit(r).candidate.key)))
            conditions = load_conditions(ROOT/input_record["file"],input_record["sha256"])
            families = tuple(f for f in paper_families() if f.name in PROTOCOL["Q3"]["families"])
            training,_ = prepare_views(conditions,families,tuple(PROTOCOL["train_conditions"]),tuple(PROTOCOL["heldout_conditions"]))
            fits = [restore_fit(r) for r in ordered]
            row.update({"selected_candidate_index":ordered[0]["job"]["candidate_index"],
                        "selected_result_file":output_path(ordered[0]["job"]).relative_to(ROOT).as_posix(),
                        "selected_result_sha256":sha(output_path(ordered[0]["job"])),
                        "selected_hypothesis":ordered[0]["fit"]["hypothesis"],
                        "training_reporting":training_diagnostics(training,fits),
                        "frozen_heldout":ordered[0]["frozen_heldout"],
                        "assessment":assess_fit(ordered[0],role)})
        else:
            row["assessment"] = {"precision_pass":False,"exact_family_topology_partition":False,"null_selected":False}
            row["training_reporting"] = {"candidate_report_tier":"failed_complete_search"}
        q3_rows.append(row)
    positives = [r for r in q3_rows if r["role"] in PROTOCOL["Q1"]["roles"]]
    structure = sum(r["complete_search"] and r["assessment"].get("exact_family_topology_partition",False) for r in positives)
    joint = sum(r["complete_search"] and r["assessment"].get("exact_family_topology_partition",False) and r["assessment"]["precision_pass"] for r in positives)
    nulls = sum(r["complete_search"] and r["assessment"].get("null_selected",False) for r in q3_rows if r["role"]=="null")
    near = sum(r["complete_search"] and r["training_reporting"]["candidate_report_tier"]=="competing_hypotheses" for r in q3_rows if r["role"]=="near_equivalent")
    q3 = {"rows":q3_rows,"positive_exact_structure":structure,"positive_joint_structure_and_precision":joint,
          "positive_denominator":9,"null_selected":nulls,"null_denominator":3,
          "near_equivalent_competing_report":near,"near_equivalent_denominator":3,
          "declared_descriptive_gate_pass":structure==9 and joint>=8 and nulls==3 and near>=2,
          "training_linear_diagnostics_CPU_seconds":time.process_time()-q3_started,
          "small_independent_qualification_not_full_multiclass":True}
    result = {"version":PROTOCOL["version"],"input_manifest_sha256":sha(OUTPUT/"input-manifest.json"),
              "source_freeze_sha256":sha(OUTPUT/"source-freeze.json"),"execution_complete_sha256":sha(OUTPUT/"execution-complete.json"),
              "fit_source_sha256":identities,"Q1":q1,"Q2":q2,"Q3":q3,
              "original_precision_eight_packets_passes":6,"original_precision_eight_packets_n":8,
              "original_results_rescored_or_rerun":False,"new_physical_integrations":0,
              "new_confirmation_program_estimates":len(records),"new_confirmation_noise_packets":inputs["new_noise_packets"],
              "cost_development_program_estimates_separate":3,"software_checks_separate":43,
              "truth_assessment_only_after_saved_fits":True,"public_release_or_submission_completed":False}
    save_json(destination,result)
    csv_path = OUTPUT/"q1_case_metrics.csv"
    with csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
        columns = ["seed","role","program","assessable","precision_pass","case_score","maximum_parameter_relative_error","maximum_amplitude_relative_error","cpu_seconds","budget_exhausted"]
        writer = csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(q1_rows)
    print(json.dumps({"Q1_wins":wins,"Q1_advantage_gate":q1["declared_conditional_advantage_gate_pass"],
                      "Q1_summary":q1["total_summary"],"Q2_partition_correct":q2["selected_correct"],
                      "Q3_structure":structure,"Q3_joint_precision":joint,"Q3_null":nulls,
                      "Q3_near_competing":near,"Q3_gate":q3["declared_descriptive_gate_pass"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
