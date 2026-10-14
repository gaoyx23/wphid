"""One development packet's oracle design audit, not a recovery experiment.

Reconstruct existing deterministic weak inputs. Truth defines the ideal atom
ONLY to test local linear information. No candidate search, fitted parameter,
decision change or new noise realization is produced.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from experiments import paper_h3_joint_family_topology_partition as h3
from wphid.discovery.conditioning import diagnose_mechanism_information

OUTPUT = ROOT / "docs/research/algorithm-evidence/development-design-information.json"


def main():
    seed, role = 42301, "independent"
    local, bases, _ = h3._assemble(seed, role)
    builder = h3._builder(local, h3.FAMILIES[h3.TRUTH_FAMILY])
    rows = []
    for equation, field in enumerate(h3.FIELDS):
        atom = builder(h3.TRUTH_CASES[role]["parameters"][equation])[equation]
        count = len(atom) // len(local)
        conditions = []
        for index, name in enumerate(h3.h1.CONDITIONS):
            block = slice(index * count, (index + 1) * count)
            conditions.append({"condition": name,
                               **diagnose_mechanism_information(bases[equation][block], atom[block])})
        rows.append({"equation": field,
                     "pooled": diagnose_mechanism_information(bases[equation], atom),
                     "conditions": conditions})
    files = (Path(__file__), ROOT / "experiments/paper_h3_joint_family_topology_partition.py",
             ROOT / "experiments/paper_h1_heterogeneous_three_class_network.py",
             ROOT / "experiments/paper_m5_two_mechanism_family_discovery.py",
             ROOT / "src/wphid/discovery/conditioning.py")
    report = {"seed": seed, "role": role, "split": "development",
              "purpose": "oracle local linear information diagnostic, NOT mechanism recovery",
              "truth_used_for": "ideal atom evaluation, never estimation or selection",
              "new_forward_integrations": 0, "new_noise_realizations": 0, "parameter_fits": 0,
              "reconstructed_existing_weak_packet": True,
              "caveat": "Fixed-atom information is not nonlinear identifiability or calibrated uncertainty. Does not separate discretization bias and errors-in-variables.",
              "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "equations": rows}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
