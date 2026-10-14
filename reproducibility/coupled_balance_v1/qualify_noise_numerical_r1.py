"""Bounded new numerical checks with prior noise-kernel identities verified."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    previous = ROOT / "artifacts/paper_coupled_balance_v1/noise_software_qualification_20261012_attempt01"
    complete = json.loads((previous / "execution-complete.json").read_text(encoding="utf-8"))
    for filename, expected in (("assessment.json", complete["assessment_sha256"]),
                               ("source-freeze.json", complete["source_freeze_sha256"])):
        if hashlib.sha256((previous / filename).read_bytes()).hexdigest() != expected:
            raise SystemExit("prior noise qualification receipt changed")
    freeze = json.loads((previous / "source-freeze.json").read_text(encoding="utf-8"))
    for item in freeze["sources"]:
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit("frozen prior noise source/test changed")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    from reproducibility.source_geometry_v6 import qualify_software as harness
    harness.PROTOCOL_PATH = ROOT / "reproducibility/coupled_balance_v1/noise-numerical-r1-protocol.json"
    harness.main()


if __name__ == "__main__":
    main()
