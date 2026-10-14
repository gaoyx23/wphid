"""Validate exact differentiation and measure its cost before confirmation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(Path(__file__).parent))

from scientific_io import load_conditions, save_json, sha, PLAN
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import Candidate, Mechanism, FitBudget
from wphid.final_candidate_v1.adam_adapter import AdamBudget, fit_adam
from wphid.final_candidate_v1.pipeline import prepare_views
from wphid.final_candidate_v1.exact_gradient import ExactGradientView


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = ROOT / "artifacts/paper_final_candidate_v1/exact_gradient_cost_development_20261011"
    if (output / "qualification.json").exists():
        raise FileExistsError("Exact-gradient cost qualification already completed")
    output.mkdir(parents=True, exist_ok=True)
    xml = output / "software-tests.xml"
    test = ROOT / "tests/final_candidate_v1/test_exact_gradient.py"
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", str(test), "-p", "no:cacheprovider",
                             "--junitxml="+str(xml), "-q"], cwd=ROOT, env=os.environ.copy(),
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        print(result.stdout); print(result.stderr)
        raise RuntimeError("Exact-gradient differentiation qualification failed")
    suites = ET.parse(xml).getroot().findall(".//testsuite")
    count = sum(int(s.get("tests",0)) for s in suites)
    original_cost = json.loads((ROOT / "artifacts/paper_final_candidate_v1/cost_development_20261011/cost.json").read_text())
    source = original_cost["input"]
    conditions = load_conditions(ROOT / source["file"], source["sha256"])
    family = next(f for f in paper_families() if f.name == "arrhenius_oxygen_damage")
    training, _ = prepare_views(conditions, (family,), tuple(PLAN["train_conditions"]), tuple(PLAN["heldout_conditions"]))
    view = ExactGradientView(training.conditions, training.scale)
    budget = AdamBudget(starts=FitBudget(seeds=(42,), initial_fractions=(.5,), maximum_cpu_seconds=60.),
                        epochs_per_start=100, maximum_backward_calls=100)
    fitted = fit_adam(view, Candidate((Mechanism("reaction", family, (0,1,2)),)), budget)
    record = {"version": "EXACT_CPU_WEAK_JACOBIAN_QUALIFICATION_V1", "software_tests_passed": count,
              "input_sha256": source["sha256"], "new_noise_observations": 0,
              "new_cost_development_estimator_fits": 1, "confirmation_accuracy_inspected": False,
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                                (ROOT / "src/wphid/final_candidate_v1/exact_gradient.py", test, Path(__file__))},
              "software_xml_sha256": sha(xml), "original_full_autograd_cost": original_cost["adam"],
              "exact_jacobian_cost": {"cpu_seconds": fitted["fit"].cpu_seconds,
                                      "wall_seconds": fitted["fit"].wall_seconds,
                                      "backward_calls": fitted["backward_calls"]},
              "same_model_and_loss": True, "precision_improvement_claim": False,
              "common_Q1_cap_cpu_seconds": 600.}
    save_json(output / "qualification.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
