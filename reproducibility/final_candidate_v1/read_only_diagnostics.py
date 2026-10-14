"""Predeclared local sensitivity/action diagnostics on saved final estimates.

No parameter optimization, observation generation, candidate reselection or
historical rescoring. These computations do not establish confidence bounds.
"""

import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
sys.path.insert(1,str(Path(__file__).parent))
sys.path.insert(2,str(ROOT))

import numpy as np
from scipy.sparse import block_diag

from scientific_io import load_conditions, save_json, sha
from run_confirmation import PROTOCOL, OUTPUT, read_json, verify_freeze
from assess_confirmation import restore_fit
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import design
from wphid.final_candidate_v1.pipeline import prepare_views


def diagnose(record, conditions):
    from experiments import paper_h3_joint_family_topology_partition as reference
    fit = restore_fit(record)
    if not fit.candidate.mechanisms:
        return {"status":"null_no_reaction_parameter_tangent"}
    mechanism = fit.candidate.mechanisms[0]
    family = mechanism.family
    families = tuple(f for f in paper_families() if f.name in PROTOCOL["Q3"]["families"])
    if record["job"]["stage"] != "Q3":
        families = (family,)
    training,heldout = prepare_views(conditions,families,tuple(PROTOCOL["train_conditions"]),
                                    tuple(PROTOCOL["heldout_conditions"]))
    if training.scale.fingerprint != fit.training_scale_sha256:
        raise ValueError("Saved tangent uses wrong scale")
    parameters = fit.candidate.split(fit.parameters)[0]
    derivatives = []
    for p in parameters:
        rows = []
        for condition in training.conditions:
            t,o,d = condition.states
            features = np.column_stack((-1/t,np.log(o),np.log1p(-d)))
            if family.name == "arrhenius_oxygen":
                features = features[:,:2]
            elif family.name == "arrhenius_damage":
                features = features[:,[0,2]]
            rate = family.values(condition.states,p)
            rows.append(np.asarray(condition.projection@(rate[:,None]*features)))
        derivatives.append(np.vstack(rows))
    matrices = design(training,fit.candidate,fit.parameters)
    nuisance = block_diag(matrices,format="csr").toarray()
    sizes = [y.size for y in training.targets]
    tangent = np.zeros((sum(sizes),len(fit.parameters)))
    width = len(family.bounds)
    span = np.asarray([hi-lo for lo,hi in family.bounds])
    offset = 0
    for e,size in enumerate(sizes):
        group = mechanism.groups[e]
        if group is not None:
            derivative = derivatives[group]*training.scale.target_weights[e]/training.scale.norms_for(family)[e]
            tangent[offset:offset+size,group*width:(group+1)*width] = derivative*fit.coefficients[e][-1]*span
        offset += size
    projected = tangent-nuisance@np.linalg.lstsq(nuisance,tangent,rcond=None)[0]
    singular = np.linalg.svd(projected,compute_uv=False)
    tolerance = np.finfo(float).eps*max(projected.shape)*singular[0] if singular.size else 0.
    rank = int(np.count_nonzero(singular>tolerance))
    column_norm = np.sum(tangent*tangent,axis=0)
    remaining = np.sum(projected*projected,axis=0)
    absorbed = 1-remaining/np.maximum(column_norm,1e-300)
    action = {}
    role = record["job"]["input"]["role"]
    if role != "null":
        true_family = next(f for f in paper_families() if f.name == reference.TRUTH_FAMILY)
        for name,view in (("training",training),("heldout",heldout)):
            errors = []
            for e in range(3):
                predicted,true = [],[]
                for condition in view.conditions:
                    raw_true = condition.atom(true_family,reference.TRUTH_CASES[role]["parameters"][e])*reference.TRUTH_AMPLITUDES[e]
                    group = mechanism.groups[e]
                    raw_predicted = np.zeros_like(raw_true)
                    if group is not None:
                        physical_amplitude = fit.coefficients[e][-1]/view.scale.norms_for(family)[e]
                        raw_predicted = condition.atom(family,parameters[group])*physical_amplitude
                    predicted.append(raw_predicted);true.append(raw_true)
                a,b = np.concatenate(predicted),np.concatenate(true)
                errors.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)))
            action[name] = errors
    return {"status":"computed","parameter_coordinates":"declared unit-box coordinates",
            "nuisance":"all equation-specific base coefficients and signed mechanism amplitudes",
            "profiled_tangent_singular_values":singular.tolist(),"numerical_rank":rank,
            "internal_parameter_dimension":len(fit.parameters),"rank_tolerance":float(tolerance),
            "smallest_to_largest_singular_ratio":float(singular[-1]/singular[0]) if singular.size and singular[0]>0 else None,
            "fraction_squared_column_norm_absorbed_by_linear_nuisance":absorbed.tolist(),
            "conditioning_on":"saved observed states, fixed weak operators and training scales at saved selected point",
            "mechanism_action_relative_errors_by_equation":action,
            "action_error_conditioning":"same observed states for estimated and reference laws; not free trajectories or field reconstruction truth",
            "calibrated_confidence_or_global_identifiability":False,
            "other_nonlinear_mechanisms_empirically_projected":False}


def main():
    verify_freeze()
    output = OUTPUT / "read-only-diagnostics.json"
    if output.exists():
        raise FileExistsError("Saved diagnostic exists; read rather than recompute")
    declaration = read_json(Path(__file__).parent / "diagnostic-declaration.json")
    if declaration["source_sha256"] != sha(Path(__file__)):
        raise ValueError("Diagnostic differs from prospective declaration")
    assessment = read_json(OUTPUT / "assessment.json")
    selected = []
    for relative in assessment["fit_source_sha256"]:
        if "/Q3/" not in relative:
            selected.append(relative)
    selected.extend(row["selected_result_file"] for row in assessment["Q3"]["rows"] if "selected_result_file" in row)
    records,cache = [],{}
    for relative in selected:
        if sha(ROOT / relative) != assessment["fit_source_sha256"][relative]:
            raise ValueError("Saved fit changed before diagnostic")
        fitted = read_json(ROOT / relative)
        if fitted["status"] != "success":
            records.append({"file":relative,"status":"failed_estimate_retained"})
            continue
        source = fitted["job"]["input"]
        if source["sha256"] not in cache:
            cache[source["sha256"]] = load_conditions(ROOT/source["file"],source["sha256"])
        records.append({"file":relative,"fit_sha256":sha(ROOT/relative),
                        "job_id":fitted["job"]["id"],"stage":fitted["job"]["stage"],
                        "role":source["role"],"seed":source["seed"],
                        "program":fitted["job"]["program"],**diagnose(fitted,cache[source["sha256"]])})
    save_json(output,{"version":"SAVED_POINT_LOCAL_SENSITIVITY_AND_ACTION_DIAGNOSTIC_V1",
                     "declaration_sha256":sha(Path(__file__).parent/"diagnostic-declaration.json"),
                     "assessment_sha256":sha(OUTPUT/"assessment.json"),"source_sha256":sha(Path(__file__)),
                     "records":records,"new_noise_observations":0,"new_parameter_fits":0,
                     "saved_estimates_reselected_or_refined":False,
                     "new_confidence_or_precision_qualification":False})
    print(json.dumps({"saved_point_diagnostics":len(records),"new_parameter_fits":0}),flush=True)


if __name__ == "__main__":
    main()
