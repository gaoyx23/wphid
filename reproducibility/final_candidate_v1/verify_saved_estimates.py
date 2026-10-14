"""Replay saved model outputs without fitting, truth scoring or new inputs."""

import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(Path(__file__).parent))

import numpy as np

from assess_confirmation import restore_fit
from run_confirmation import OUTPUT, PROTOCOL, jobs_for, output_path, read_json, verify_freeze
from scientific_io import load_conditions, save_json, sha
from wphid.final_candidate_v1.pipeline import prepare_views
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import ScoreConvention, design, physical_coefficients, predict_frozen


def require_close(actual, expected, label, *, rtol=5e-8, atol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Invalid replay shape/value: " + label)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError("Saved output cannot be replayed: " + label)
    return float(np.max(np.abs(actual-expected), initial=0.))


def replay(record, training, heldout):
    fit = restore_fit(record)
    if training.scale.fingerprint != fit.training_scale_sha256 or record["training_scale_sha256"] != fit.training_scale_sha256:
        raise ValueError("Training scale identity changed")
    bounds = np.asarray(fit.candidate.flat_bounds, dtype=float).reshape(-1, 2)
    if fit.parameters.shape != (len(bounds),) or not np.isfinite(fit.parameters).all():
        raise ValueError("Invalid saved nonlinear parameter vector")
    if np.any(fit.parameters < bounds[:,0]-1e-10) or np.any(fit.parameters > bounds[:,1]+1e-10):
        raise ValueError("Saved nonlinear parameters escaped declared bounds")
    for mechanism, parameters, saved in zip(fit.candidate.mechanisms, fit.candidate.split(fit.parameters), record["fit"]["hypothesis"]):
        require_close(parameters, saved["internal_parameters_by_group"], "hypothesis parameter transcription")
    matrices = design(training, fit.candidate, fit.parameters)
    rss = tuple(float(np.sum((y-a@c)**2)) for a,y,c in zip(matrices,training.targets,fit.coefficients))
    residual_difference = require_close(rss, fit.equation_rss, "equation training RSS")
    count = fit.candidate.parameter_count(training.bases)
    if count != fit.nominal_parameter_count:
        raise ValueError("Nominal parameter count changed")
    convention = ScoreConvention.from_targets(training.targets)
    score_difference = require_close(convention.score(rss,count), fit.final_score, "common final score")
    coefficient_difference = max(require_close(a,b,"physical coefficient units") for a,b in
                                 zip(physical_coefficients(training,fit),record["fit"]["physical_coefficients"]))
    predictions = predict_frozen(heldout,fit)
    targets = tuple(np.concatenate([c.targets[e] for c in heldout.conditions]) for e in range(len(predictions)))
    relative = [float(np.linalg.norm(y-p)/max(np.linalg.norm(y),1e-12)) for y,p in zip(targets,predictions)]
    heldout_difference = require_close(relative,record["frozen_heldout"]["equation_weak_target_relative_residual"],
                                      "frozen held-out weak residual")
    if record["frozen_heldout"]["refitted"] or record["truth_used_in_fitting"] or record["heldout_used_in_fitting"]:
        raise ValueError("Saved estimate violates fitting information boundary")
    objective_scope = "profile log-RSS objective independently replayed"
    if record["job"]["program"] == "wphid":
        require_close(convention.objective(rss),fit.training_objective,"WPHID own objective")
    else:
        terminals = [r["terminal_training_objective"] for r in fit.restarts]
        require_close(min(terminals),fit.training_objective,"ADAM terminal restart selection")
        mse = sum(r/n for r,n in zip(rss,convention.row_counts))
        if fit.training_objective + 1e-10 < mse:
            raise ValueError("Recorded ADAM objective is below its nonnegative-penalty lower bound")
        objective_scope = "recorded restart selection and residual term checked; terminal Gamma penalty cannot be replayed"
    return {"status":"saved_model_replay_passed", "nonlinear_parameter_dimension":len(fit.parameters),
            "nominal_parameter_count":count,"training_RSS_max_absolute_difference":residual_difference,
            "common_score_absolute_difference":score_difference,
            "physical_coefficient_max_absolute_difference":coefficient_difference,
            "heldout_residual_max_absolute_difference":heldout_difference,"objective_replay_scope":objective_scope}


def main():
    inputs = verify_freeze()
    assessment = read_json(OUTPUT/"assessment.json")
    read_json(OUTPUT/"execution-complete.json")
    destination = OUTPUT/"saved-estimate-verification.json"
    if destination.exists():
        raise FileExistsError("Saved-output verification exists; read it instead of overwriting")
    families = tuple(f for f in paper_families() if f.name in PROTOCOL["Q3"]["families"])
    cache, records, identities = {}, [], {}
    for stage in ("Q1","Q2","Q3"):
        for job in jobs_for(stage,inputs):
            path = output_path(job)
            relative = path.relative_to(ROOT).as_posix()
            expected = assessment["fit_source_sha256"][relative]
            if sha(path) != expected:
                raise ValueError("Saved fit changed before output replay: " + relative)
            record = read_json(path)
            if record["job"] != job or not record["completed"]:
                raise ValueError("Saved fit/job identity differs")
            identities[relative] = expected
            result = {"file":relative,"fit_sha256":expected,"job_id":job["id"],"stage":stage}
            if record["status"] != "success":
                result["status"] = "failed_estimate_retained_without_fabricated_predictions"
            else:
                key = job["input"]["sha256"]
                if key not in cache:
                    conditions = load_conditions(ROOT/job["input"]["file"],key)
                    applicable = families if stage == "Q3" else tuple(f for f in families if f.name == PROTOCOL["Q1"]["known_family"])
                    cache[key] = prepare_views(conditions,applicable,tuple(PROTOCOL["train_conditions"]),tuple(PROTOCOL["heldout_conditions"]))
                result.update(replay(record,*cache[key]))
            records.append(result)
    if len(records) != 789 or identities != assessment["fit_source_sha256"]:
        raise ValueError("Saved-output replay did not cover the declared complete fit set")
    save_json(destination,{"version":"saved-output-replay-20261011-v1","source_freeze_sha256":sha(OUTPUT/"source-freeze.json"),
               "assessment_sha256":sha(OUTPUT/"assessment.json"),"verifier_source_sha256":sha(__file__),
               "all_789_record_identities_checked":True,"successful_model_replays":sum(r["status"]=="saved_model_replay_passed" for r in records),
               "failed_estimates_retained":sum(r["status"]!="saved_model_replay_passed" for r in records),
               "relative_tolerance":5e-8,"absolute_tolerance":1e-10,"records":records,
               "new_inputs":0,"new_optimizer_calls":0,"coefficient_refits":0,"new_truth_assessments":0,
               "ADAM_terminal_Gamma_archived":False,"full_optimizer_checkpoint_reproduction_claimed":False})
    print(json.dumps({"saved_estimate_records_verified":len(records),"new_optimizer_calls":0}),flush=True)


if __name__ == "__main__":
    main()
