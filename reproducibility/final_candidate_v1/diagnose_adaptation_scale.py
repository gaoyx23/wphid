"""Assess fixed-threshold geometry after saved estimates; never tune or fit."""

import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(1,str(Path(__file__).parent))
sys.path.insert(2,str(ROOT))

import numpy as np
from run_confirmation import OUTPUT, PROTOCOL, read_json, verify_freeze
from scientific_io import load_conditions, save_json, sha
from wphid.final_candidate_v1.pipeline import prepare_views
from wphid.final_candidate_v1.problem import paper_families


def main():
    verify_freeze()
    assessment=read_json(OUTPUT/"assessment.json")
    destination=OUTPUT/"adaptation-scale-diagnostic.json"
    if destination.exists():
        raise FileExistsError("Adaptation diagnostic exists; read rather than overwrite")
    declaration=read_json(Path(__file__).parent/"adaptation-scale-declaration.json")
    if declaration["source_sha256"] != sha(__file__):
        raise ValueError("Diagnostic differs from its pre-assessment declaration")
    from experiments import paper_h3_joint_family_topology_partition as reference
    family=next(f for f in paper_families() if f.name==PROTOCOL["Q1"]["known_family"])
    threshold=PROTOCOL["adam_budget"]["amplitude_threshold"]
    tolerance=PROTOCOL["precision_gates"]["active_reaction_amplitude"]
    amplitude=np.abs(np.asarray(reference.TRUTH_AMPLITUDES))
    records=[]
    for relative,expected in assessment["fit_source_sha256"].items():
        if "/Q1/" not in relative:
            continue
        fit=read_json(ROOT/relative)
        if fit["job"]["program"]!="adam":
            continue
        if sha(ROOT/relative)!=expected:
            raise ValueError("Saved ADAM estimate changed")
        source=fit["job"]["input"]
        conditions=load_conditions(ROOT/source["file"],source["sha256"])
        training,_=prepare_views(conditions,(family,),tuple(PROTOCOL["train_conditions"]),tuple(PROTOCOL["heldout_conditions"]))
        norms=np.asarray(training.scale.norms_for(family))
        true_scaled=amplitude*norms
        upper=true_scaled*(1+tolerance)
        impossible=upper < threshold
        coefficients=fit.get("fit",{}).get("scaled_coefficients")
        observed_zero=[abs(c[-1])<=1e-12 for c in coefficients] if coefficients is not None else None
        records.append({"file":relative,"fit_sha256":expected,"seed":source["seed"],"role":source["role"],
                        "training_scale_sha256":training.scale.fingerprint,
                        "true_amplitude_magnitudes_in_training_scaled_units":true_scaled.tolist(),
                        "threshold_in_physical_amplitude_units":(threshold/norms).tolist(),
                        "upper_25percent_acceptable_scaled_amplitude":upper.tolist(),
                        "precision_band_excluded_by_fixed_threshold":impossible.tolist(),
                        "any_amplitude_gate_impossible":bool(np.any(impossible)),
                        "saved_zero_amplitudes":observed_zero})
    if len(records)!=54:
        raise ValueError("Diagnostic must retain all 54 ADAM estimates")
    save_json(destination,{"version":"PREDECLARED_FIXED_THRESHOLD_GEOMETRY_DIAGNOSTIC_V1",
              "declaration_sha256":sha(Path(__file__).parent/"adaptation-scale-declaration.json"),
              "source_sha256":sha(__file__),"assessment_sha256":sha(OUTPUT/"assessment.json"),
              "fixed_scaled_amplitude_threshold":threshold,"reaction_amplitude_error_gate":tolerance,
              "records":records,"amplitude_gate_impossible_estimates":sum(r["any_amplitude_gate_impossible"] for r in records),
              "interpretation":"A zero or above-threshold coefficient cannot enter an amplitude acceptance band wholly below threshold. This diagnoses the fixed adaptation, not native ADAM-SINDy or WPHID superiority.",
              "new_parameter_fits":0,"new_inputs":0,"frozen_primary_gate_changed":False,
              "threshold_or_penalty_tuned_on_confirmation":False})
    print(json.dumps({"adaptation_scale_records":len(records),"new_parameter_fits":0}),flush=True)


if __name__=="__main__":
    main()
