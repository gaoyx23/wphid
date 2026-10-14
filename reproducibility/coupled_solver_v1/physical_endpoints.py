"""Same saved-fit physical coordinates across different parameterizations.

Supplemental completeness audit, not a replacement/tuning of the frozen gate.
Absent paths and missing kinetic coordinates cannot become precision successes.
No fit, restart, candidate, weight, data or threshold is changed.
"""

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]


def physical_coordinates(source,point,coefficients):
    from wphid.coupled_solver_v1.model import active_paths
    result={}
    paths=active_paths(source)
    for path in ("oxygen","steam"):
        for field in ("carbon","pO2" if path=="oxygen" else "pH2O","pProducts","T"):
            key=f"{field}.{path}"
            if path not in paths:
                result[key+".response"]=0.;continue
            if source in ("both_free","both_independent"):
                amplitude=coefficients.get(key+"_response")
            elif field=="carbon":
                amplitude=coefficients.get("carbon_rate_scale")
                if path=="steam" and len(paths)==2:amplitude=None if amplitude is None else amplitude*point["reaction.relative_rate"]
            elif field in ("pO2","pH2O"):
                amplitude=coefficients.get(path+"_response")
            else:
                name="product_response" if field=="pProducts" else "heat_response"
                amplitude=coefficients.get(name)
                if source=="both_oxygen_heat" and field=="T" and path=="steam":amplitude=0.
                elif len(paths)==2:
                    ratio=point["reaction.relative_rate"]
                    if (source=="both_swapped_product" and field=="pProducts" and path=="oxygen") or (path=="steam" and not(source=="both_swapped_product" and field=="pProducts")):
                        amplitude=None if amplitude is None else amplitude*ratio
            result[key+".response"]=amplitude
            if amplitude==0.:continue
            prefix=f"kinetic.{key}." if source=="both_independent" else f"kinetic.{path}."
            result[key+".energy_1e5"]=point.get(prefix+"energy_1e5")
            result[key+".pressure_order"]=point.get(prefix+"pressure_order")
            result[key+".carbon_order"]=point.get(prefix+"carbon_order" if source=="both_independent" else "kinetic.carbon_order")
    return result


def grade(source,fit,truth,precision):
    if not fit.get("fitted"):
        return {"assessable":False,"complete_physical_parameter_pass":False,"missing_fit":True}
    target=physical_coordinates(truth["source"],truth["parameters"],truth["coefficients"])
    actual=physical_coordinates(source,fit["parameters"],fit["coefficients"])
    errors={};missing=[];ratios=[];zero_violations=[]
    for key,value in target.items():
        estimate=actual.get(key)
        if estimate is None:
            missing.append(key);continue
        if value==0.:
            if abs(estimate)>1e-12:zero_violations.append({"key":key,"absolute_response":abs(estimate)})
            continue
        relative=abs(estimate-value)/abs(value)
        threshold=precision["active_source_response_relative" if key.endswith("response") else
            ("activation_energy_relative" if key.endswith("energy_1e5") else ("pressure_order_relative" if key.endswith("pressure_order") else "carbon_order_relative"))]
        errors[key]=relative;ratios.append(relative/threshold)
    score=max(ratios) if ratios else None
    return {"assessable":bool(ratios and not missing),"complete_physical_parameter_pass":bool(ratios and not missing and not zero_violations and score<=1.),
        "reaction_score":score,"missing_coordinates":missing,"zero_path_violations":zero_violations,
        "physical_relative_errors":errors,"coordinates_replicated_across_destinations_are_not_independent_parameters":True}


def main():
    from reproducibility.coupled_solver_v1.workflow import sha,write,verify_freeze
    protocol=json.loads((ROOT/"reproducibility/coupled_solver_v1/scientific-protocol-20261013.json").read_text(encoding="utf-8"))
    output=ROOT/protocol["output_root"];verify_freeze(output,protocol)
    if (output/"physical-endpoint-audit.json").exists():
        raise SystemExit("completed endpoint audit retained; do not rescore")
    fits_receipt=json.loads((output/"fits-complete.json").read_text(encoding="utf-8"))
    if any(sha(output/"fits"/i["path"])!=i["sha256"] for i in fits_receipt["files"]):raise SystemExit("saved fits changed")
    assessment=json.loads((output/"assessment.json").read_text(encoding="utf-8"));truth=json.loads((output/"truth_scoring_only/truth.json").read_text(encoding="utf-8"))
    records=[];pairs=[]
    for selected in assessment["selected_records"]:
        block=selected["block"];program=selected["program"]
        if selected["selected"] is None:
            records.append({"block":block,"program":program,"role":selected["role"],"complete_physical_parameter_pass":False});continue
        matches=[]
        for path in (output/"fits").glob(block+"_*_"+program+".json"):
            record=json.loads(path.read_text(encoding="utf-8"))
            if record["candidate"]==selected["selected"]:matches.append(record)
        if len(matches)!=1:raise SystemExit("selected fit identity not unique")
        fit=matches[0]["fit"]
        records.append({"block":block,"program":program,"role":selected["role"],"selected":selected["selected"],
            **grade(selected["selected"]["source"],fit,truth[block],protocol["precision"])})
    for oldpair in assessment["source_component_pairs"]:
        block=oldpair["block"];fits={}
        for path in (output/"fits").glob(block+"_*_wphid.json"):
            record=json.loads(path.read_text(encoding="utf-8"));c=record["candidate"]
            if c["source"] in ("both_shared","both_free") and c["gas"]=="combined_opening" and c["thermal"]=="stiffness" and c["flux"]=="Darcy":fits[c["source"]]=record["fit"]
        pairs.append({"block":block,**{source:grade(source,fit,truth[block],protocol["precision"]) for source,fit in fits.items()}})
    counts={program:sum(r.get("complete_physical_parameter_pass",False) for r in records if r["program"]==program and r["role"] not in ("null","near_equivalent")) for program in ("wphid","adam")}
    write(output/"physical-endpoint-audit.json",{"identity":"supplemental_same_saved_fit_physical_coordinates_and_completeness",
        "original_frozen_adoption_gate_replaced":False,"new_fits":0,"new_noise_blocks":0,"new_trajectories":0,
        "source_sha256":sha(Path(__file__)),"fits_receipt_sha256":sha(output/"fits-complete.json"),
        "positive_complete_physical_parameter_counts":counts,"positive_denominator_per_program":12,"selected":records,"source_component_pairs":pairs})
    print(json.dumps({"physical_parameter_counts":counts,"positive_denominator":12,"new_fits":0}))


if __name__=="__main__":main()
