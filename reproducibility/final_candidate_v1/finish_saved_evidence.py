"""Continue evidence preparation after the authorized fixed run finishes.

This follower never starts/refits estimators or regenerates inputs. It reads
completed records and runs only their assessment, diagnostics and packaging.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((Path(__file__).parent/"scientific-protocol.json").read_text())
SCI = ROOT/PROTOCOL["output_root"]
OUTPUT = ROOT/"artifacts/paper_final_candidate_v1/evidence_preparation_20261011"


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Wrong worktree")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    completed = SCI/"execution-complete.json"
    while not completed.exists():
        time.sleep(5)
    steps = [
        ("assess_saved_estimates",Path(__file__).parent/"assess_confirmation.py",SCI/"assessment.json"),
        ("read_only_local_diagnostics",Path(__file__).parent/"read_only_diagnostics.py",SCI/"read-only-diagnostics.json"),
        ("local_release_candidate",Path(__file__).parent/"prepare_release_candidate.py",ROOT/"artifacts/paper_final_candidate_v1/release_candidate_20261011/archive-identity.json"),
        ("paper_tables_and_comparison_figure",ROOT/"docs/paper/manuscript/ress-20261010/prepare_new_confirmation_evidence.py",ROOT/"docs/paper/manuscript/ress-20261010/assets/final-candidate-scientific-evidence.json")
    ]
    records = []
    for name,script,marker in steps:
        if marker.exists():
            records.append({"step":name,"completed_record_reused":True})
        else:
            result = subprocess.run([sys.executable,"-B",str(script)],cwd=ROOT,env=os.environ.copy())
            if result.returncode != 0 or not marker.exists():
                (OUTPUT/"status.json").write_text(json.dumps({"step":name,"status":"needs_inspection", "exit_code":result.returncode,"completed_steps":records},indent=2)+"\n")
                raise RuntimeError("Evidence preparation failed at "+name)
            records.append({"step":name,"completed_record_reused":False})
        records[-1]["result_sha256"] = hashlib.sha256(marker.read_bytes()).hexdigest()
        (OUTPUT/"status.json").write_text(json.dumps({"status":"progressing","completed_steps":records},indent=2)+"\n")
        print(json.dumps({"evidence_step_completed":name}),flush=True)
    (OUTPUT/"complete.json").write_text(json.dumps({"status":"scientific_evidence_and_local_candidate_prepared",
                       "steps":records,"new_estimator_fits_or_input_generation_in_this_follower":0,
                       "manuscript_source_review_and_final_exports":"follow after evidence inspection"},indent=2)+"\n")
    print(json.dumps({"evidence_preparation_completed":True,"new_estimator_fits":0}),flush=True)


if __name__ == "__main__":
    main()
