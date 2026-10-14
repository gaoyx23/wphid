"""Only the new sparse-observation checks; preserved dependencies are verified."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/coupled_observation_v1/software-protocol-20261013.json"


def verify_dependencies(protocol):
    for item in protocol["required_completed_records"]:
        folder = ROOT / item["path"]
        receipt = json.loads((folder / "execution-complete.json").read_text(encoding="utf-8"))
        for filename, expected in (("assessment.json", item["assessment_sha256"]),
                (item["freeze_filename"], receipt[item["freeze_receipt_key"]])):
            if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != expected:
                raise SystemExit("required completed record changed; no rerun")
        if receipt["status"] != "completed" or not receipt[item["passed_key"]]:
            raise SystemExit("required completed record not qualified")
        freeze = json.loads((folder / item["freeze_filename"]).read_text(encoding="utf-8"))
        for source in freeze.get("sources", freeze.get("source_files", [])):
            if hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise SystemExit("frozen dependency source changed")


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    verify_dependencies(protocol)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    from reproducibility.source_geometry_v6 import qualify_software as harness
    harness.PROTOCOL_PATH = PROTOCOL_PATH
    harness.main()


if __name__ == "__main__":
    main()
