"""Preserve both failed references and qualify only the new cusp-aware checks."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/coupled_observation_v1/derivative-resolution-r2-protocol-20261013.json"


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for record in protocol["retained_attempts"]:
        previous = ROOT / record["path"]
        complete = json.loads((previous / "execution-complete.json").read_text(encoding="utf-8"))
        if complete["status"] != "completed":
            raise SystemExit("prior attempt is not complete")
        for filename, expected in (("assessment.json", record["assessment_sha256"]),
                ("source-freeze.json", complete["source_freeze_sha256"])):
            if hashlib.sha256((previous / filename).read_bytes()).hexdigest() != expected:
                raise SystemExit("prior failed record changed")
        freeze = json.loads((previous / "source-freeze.json").read_text(encoding="utf-8"))
        for item in freeze["sources"]:
            if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise SystemExit("prior kernel, protocol or tests changed")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    from reproducibility.coupled_observation_v1.qualify_software import verify_dependencies
    verify_dependencies(protocol)
    from reproducibility.source_geometry_v6 import qualify_software as harness
    harness.PROTOCOL_PATH = PROTOCOL_PATH
    harness.main()


if __name__ == "__main__":
    main()
