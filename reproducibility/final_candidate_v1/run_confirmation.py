"""Checkpointed Q1 -> Q2 -> Q3 execution; truth assessment is separate.

All completed success/failure records are reused by their identities. This
runner cannot change frozen sources or regenerate observations.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import importlib.metadata
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

from scientific_io import load_conditions, save_json, sha
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import Candidate, Mechanism, FitBudget, fit_candidate, enumerate_candidates
from wphid.final_candidate_v1.adam_adapter import AdamBudget, fit_adam
from wphid.final_candidate_v1.pipeline import prepare_views, fit_record, frozen_residual_report
from wphid.final_candidate_v1.exact_gradient import ExactGradientView

PROTOCOL_PATH = Path(__file__).parent / "scientific-protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
OUTPUT = ROOT / PROTOCOL["output_root"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_freeze():
    import wphid
    if Path.cwd().resolve() != ROOT or not Path(wphid.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("Wrong working directory or editable import")
    source = read_json(OUTPUT / "source-freeze.json")
    for relative, expected in {**source["source_sha256"], **source["frozen_reference_sha256"]}.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    inputs = read_json(OUTPUT / "input-manifest.json")
    if inputs["source_freeze_sha256"] != sha(OUTPUT / "source-freeze.json"):
        raise ValueError("Input manifest is not bound to source freeze")
    runtime = read_json(Path(__file__).parent / "runtime-lock.json")
    for distribution, expected in runtime["numerical_test_dependency_closure"].items():
        if importlib.metadata.version(distribution) != expected:
            raise ValueError("Runtime dependency changed: " + distribution)
    return inputs


def initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify_freeze()


def output_path(job):
    return OUTPUT / "fits" / job["stage"] / (job["id"]+".json")


def run_job(job):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    path = output_path(job)
    if path.exists():
        old = read_json(path)
        if old["job"] != job or old["input_sha256"] != job["input"]["sha256"]:
            raise ValueError("Existing fit differs from declared job")
        return {"id": job["id"], "reused": True, "status": old["status"]}
    conditions = load_conditions(ROOT / job["input"]["file"], job["input"]["sha256"])
    families = tuple(f for f in paper_families() if f.name in PROTOCOL["Q3"]["families"])
    if job["stage"] == "Q3":
        candidate = tuple(enumerate_candidates({"reaction": families}, 3))[job["candidate_index"]]
    else:
        family = next(f for f in families if f.name == PROTOCOL["Q1"]["known_family"])
        families = (family,)
        candidate = Candidate((Mechanism("reaction", family, tuple(job["groups"])),))
    training, heldout = prepare_views(conditions, families, tuple(PROTOCOL["train_conditions"]),
                                     tuple(PROTOCOL["heldout_conditions"]))
    setup_cpu = time.process_time()-started_cpu
    record = {"job": job, "input_sha256": job["input"]["sha256"],
              "source_freeze_sha256": sha(OUTPUT / "source-freeze.json"),
              "input_manifest_sha256": sha(OUTPUT / "input-manifest.json"),
              "training_scale_sha256": training.scale.fingerprint,
              "actual_import": str(Path(sys.modules["wphid"].__file__).resolve()),
              "threads": 1, "device": "CPU", "dtype": "float64",
              "shared_input_loading_and_scaling_cpu_seconds": setup_cpu,
              "heldout_used_in_fitting": False, "truth_used_in_fitting": False,
              "scientific_result_identity": "new confirmation estimate, not software fixture",
              "completed": True}
    try:
        if job["program"] == "adam":
            settings = PROTOCOL["adam_budget"]
            starts = FitBudget(seeds=tuple(settings["seeds"]), initial_fractions=tuple(settings["initial_fractions"]),
                               maximum_cpu_seconds=settings["maximum_cpu_seconds"])
            budget = AdamBudget(starts=starts, **{k: settings[k] for k in
                               ("epochs_per_start", "maximum_backward_calls", "learning_rate", "scheduler_step",
                                "scheduler_multiplier", "amplitude_threshold", "sparsity_multiplier")})
            result = fit_adam(ExactGradientView(training.conditions, training.scale), candidate, budget)
            fitted = result.pop("fit")
            record["adam_adaptation"] = result
            record["budget"] = asdict(budget)
        else:
            settings = dict(PROTOCOL["wphid_budget"])
            settings["seeds"] = tuple(settings["seeds"])
            settings["initial_fractions"] = tuple(settings["initial_fractions"])
            if job["stage"] == "Q3":
                settings["maximum_cpu_seconds"] = PROTOCOL["Q3"]["budget_per_candidate"]["maximum_cpu_seconds"]
            budget = FitBudget(**settings)
            fitted = fit_candidate(training, candidate, budget)
            record["budget"] = asdict(budget)
        record.update({"status": "success", "fit": fit_record(training, fitted),
                       "flat_parameters": fitted.parameters.tolist(),
                       "frozen_heldout": frozen_residual_report(heldout, fitted)})
    except (RuntimeError, FloatingPointError, ValueError) as error:
        record.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                       "failure_retained_in_denominator": True})
    record["total_worker_job_cpu_seconds"] = time.process_time()-started_cpu
    record["total_worker_job_wall_seconds"] = time.perf_counter()-started_wall
    record["common_cap_guarantees_convergence"] = False
    if path.exists():
        raise FileExistsError("Refusing to overwrite a completed scientific estimate")
    save_json(path, record)
    return {"id": job["id"], "reused": False, "status": record["status"],
            "cpu_seconds": record["total_worker_job_cpu_seconds"],
            "optimizer": record.get("fit", {}).get("optimizer_status")}


def jobs_for(stage, inputs):
    records = [r for r in inputs["records"] if r["stage"] == ("Q3" if stage == "Q3" else "Q1")]
    jobs = []
    for record in records:
        seed, role = record["seed"], record["role"]
        if stage == "Q1":
            for program in ("wphid", "adam"):
                jobs.append({"id": str(seed)+"_"+role+"_"+program, "stage": stage,
                             "input": record, "program": program,
                             "groups": PROTOCOL["Q1"]["known_partition_by_role"][role]})
        elif stage == "Q2":
            if seed not in PROTOCOL["Q2"]["reuse_Q1_block_seeds"]:
                continue
            for partition, groups in PROTOCOL["Q2"]["partitions"].items():
                if partition == role:
                    continue
                jobs.append({"id": str(seed)+"_"+role+"_as_"+partition, "stage": stage,
                             "input": record, "program": "wphid", "groups": groups,
                             "partition_name": partition})
        else:
            for index in range(PROTOCOL["Q3"]["candidate_count_including_null"]):
                jobs.append({"id": str(seed)+"_"+role+"_candidate_"+str(index).zfill(2),
                             "stage": stage, "input": record, "program": "wphid", "candidate_index": index})
    return jobs


def main():
    inputs = verify_freeze()
    status_path = OUTPUT / "execution-status.json"
    if (OUTPUT / "execution-complete.json").exists():
        raise FileExistsError("Confirmation complete; assess saved estimates, never rerun")
    started, stage_records = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=PROTOCOL["concurrency"]["worker_processes"], initializer=initialize_worker) as executor:
        for stage in ("Q1", "Q2", "Q3"):
            jobs = jobs_for(stage, inputs)
            futures = {executor.submit(run_job, job): job for job in jobs}
            completed = []
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                save_json(status_path, {"stage": stage, "completed_stage_jobs": len(completed),
                                       "total_stage_jobs": len(jobs), "completed_prior_stages": stage_records,
                                       "wall_seconds_this_invocation": time.perf_counter()-started,
                                       "last_completed": result, "assessment_not_yet_run": True})
                if len(completed) % 6 == 0 or len(completed) == len(jobs):
                    print(json.dumps({"stage": stage, "completed": len(completed), "total": len(jobs),
                                      "last": result}, ensure_ascii=False), flush=True)
            stage_records.append({"stage": stage, "jobs": len(jobs),
                                  "new_estimates": sum(not r["reused"] for r in completed),
                                  "reused_completed_estimates": sum(r["reused"] for r in completed),
                                  "failures_retained": sum(r["status"] == "failed" for r in completed),
                                  "job_cpu_seconds": sum(read_json(output_path(job))["total_worker_job_cpu_seconds"] for job in jobs)})
            print(json.dumps({"stage_completed": stage, "record": stage_records[-1]}, ensure_ascii=False), flush=True)
    save_json(OUTPUT / "execution-complete.json", {"version": PROTOCOL["version"],
              "source_freeze_sha256": sha(OUTPUT / "source-freeze.json"),
              "input_manifest_sha256": sha(OUTPUT / "input-manifest.json"),
              "stages": stage_records, "wall_seconds_this_invocation": time.perf_counter()-started,
              "new_physical_integrations": 0, "truth_accuracy_assessed": False,
              "completed_estimates_never_rerun": True})
    print(json.dumps({"execution_completed": True, "stages": stage_records}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
