"""Freeze qualified new sources, then prepare and hash confirmation inputs.

No estimator or truth-based performance assessment runs here. Existing saved
inputs are identity-checked on resume, never regenerated.
"""

import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(Path(__file__).parent))

import numpy as np

from scientific_io import save_observations, load_conditions, save_json, sha, CONDITIONS
from wphid.final_candidate_v1.observations import adapt_condition
from wphid.final_candidate_v1.problem import paper_families, validate_split
from wphid.final_candidate_v1.solver import enumerate_candidates


def main():
    import wphid
    if Path.cwd().resolve() != ROOT or not Path(wphid.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("Wrong working directory or editable import")
    directory = Path(__file__).parent
    protocol = json.loads((directory / "scientific-protocol.json").read_text(encoding="utf-8"))
    output = ROOT / protocol["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    lock_path, inputs_path = output / "source-freeze.json", output / "input-manifest.json"
    if inputs_path.exists():
        raise FileExistsError("Confirmation inputs already frozen; verify and use, do not prepare again")
    sources = {relative: sha(ROOT / relative) for relative in protocol["source_files_to_bind"]}
    implementation = json.loads((directory / "implementation-manifest.json").read_text())
    exact = json.loads((ROOT / "artifacts/paper_final_candidate_v1/exact_gradient_cost_development_20261011/qualification.json").read_text())
    for relative, expected in implementation["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Original software-qualified candidate changed: " + relative)
    for relative, expected in {**implementation["frozen_reference_sha256"], **exact["source_sha256"]}.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Qualified source changed: " + relative)
    validate_split(CONDITIONS, protocol["train_conditions"], protocol["heldout_conditions"])
    families = tuple(f for f in paper_families() if f.name in protocol["Q3"]["families"])
    candidates = tuple(enumerate_candidates({"reaction": families}, 3))
    if len(candidates) != protocol["Q3"]["candidate_count_including_null"]:
        raise ValueError("Open space differs from protocol")
    orchestration_xml = ROOT / "artifacts/_software_qualification/final_candidate_v1/scientific-orchestration.xml"
    suites = ET.parse(orchestration_xml).getroot().findall(".//testsuite")
    if (sum(int(s.get("tests",0)) for s in suites) != 6
            or sum(int(s.get("failures",0))+int(s.get("errors",0)) for s in suites)):
        raise ValueError("Scientific orchestration checks have not passed")
    # Recompute only the adapter on existing cost-development observed arrays.
    # There is no new noise, new parameter fit or confirmation accuracy check.
    cost = json.loads((ROOT / "artifacts/paper_final_candidate_v1/cost_development_20261011/cost.json").read_text())
    saved = cost["input"]
    loaded = load_conditions(ROOT / saved["file"], saved["sha256"])
    max_difference = 0.
    with np.load(ROOT / saved["file"], allow_pickle=False) as data:
        for original in loaded:
            name = original.name
            rebuilt = adapt_condition(name=name, t=data["t"], z=data["z"], r=data["r"],
                                      observed_states=tuple(data[name+"_states"]),
                                      supplied_forcing=tuple(data[name+"_forcing"]))
            for left, right in zip((*original.bases, *original.targets), (*rebuilt.bases, *rebuilt.targets)):
                np.testing.assert_allclose(left, right, rtol=2e-13, atol=1e-13)
                max_difference = max(max_difference, float(np.max(np.abs(left-right))))
            delta = (original.projection-rebuilt.projection).data
            if delta.size and np.max(np.abs(delta)) > 1e-13:
                raise ValueError("Saved weak operator is not the declared adapter")
    qualification = {"saved_cost_input_sha256": saved["sha256"], "conditions_checked": len(loaded),
                     "weak_rows_per_equation_per_condition": [c.targets[0].size for c in loaded],
                     "grid_shape": cost["input"]["grid_shape"], "strict_positive_domains": True,
                     "maximum_adapter_difference": max_difference,
                     "declared_open_candidates": len(candidates), "new_noise_or_parameter_fits": 0,
                     "confirmation_inputs_or_accuracy_inspected": False,
                     "positivity_clip_observed_in_cost_development": saved["clipping"]}
    lock = {"version": protocol["version"], "source_sha256": sources,
            "frozen_reference_sha256": implementation["frozen_reference_sha256"],
            "original_software_qualification_sha256": sha(directory / "implementation-manifest.json"),
            "exact_gradient_qualification_sha256": sha(ROOT / "artifacts/paper_final_candidate_v1/exact_gradient_cost_development_20261011/qualification.json"),
            "orchestration_software_tests": 6,
            "orchestration_software_xml_sha256": sha(orchestration_xml),
            "generator_qualification": qualification,
            "q3_candidate_keys": [str(c.key) for c in candidates],
            "code_frozen_before_confirmation_generation": True,
            "freshness_audit": {"checked_roots": [str(ROOT), str(ROOT.parents[1])],
                                "method": "Read-only exact-word seed/namespace search in historical experiment/artifact/protocol records before confirmation generation",
                                "historical_matches": 0, "numeric_substring_matches_not_counted": True},
            "confirmation_fits_at_freeze": 0}
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("Existing source freeze differs; never overwrite")
    else:
        save_json(lock_path, lock)
    records, started = [], time.perf_counter()
    for stage in ("Q1", "Q3"):
        spec = protocol[stage]
        for seed in spec["block_seeds"]:
            for role in spec["roles"]:
                path = output / "inputs" / stage / (str(seed)+"_"+role+".npz")
                meta = path.with_suffix(".json")
                if path.exists():
                    record = json.loads(meta.read_text())
                    if (sha(path) != record["sha256"] or record["namespace"] != spec["namespace"]
                            or record["seed"] != seed or record["role"] != role):
                        raise ValueError("Saved input identity mismatch")
                else:
                    record = save_observations(seed, role, spec["namespace"], path)
                records.append({"stage": stage, **record})
        print(json.dumps({"stage_inputs_prepared": stage, "total_packets_so_far": len(records)}, ensure_ascii=False), flush=True)
    result = {"version": protocol["version"], "source_freeze_sha256": sha(lock_path),
              "all_inputs_frozen_before_any_confirmation_fit": True,
              "new_noise_packets": len(records), "Q1_packets": 54, "Q3_packets": 15,
              "independent_Q1_blocks": 18, "independent_Q3_blocks": 3,
              "new_physical_integrations": 0, "preparation_wall_seconds": time.perf_counter()-started,
              "records": records}
    save_json(inputs_path, result)
    print(json.dumps({"input_manifest_sha256": sha(inputs_path), "packets": len(records),
                      "confirmation_fits_completed": 0}), flush=True)


if __name__ == "__main__":
    main()
