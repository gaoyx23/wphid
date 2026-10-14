"""New likelihood checks only, with verified frozen dependencies."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/coupled_likelihood_v1/software-protocol-20261013.json"


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    from reproducibility.coupled_observation_v1.qualify_software import verify_dependencies
    verify_dependencies(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
    from reproducibility.source_geometry_v6 import qualify_software as harness
    harness.PROTOCOL_PATH = PROTOCOL_PATH
    harness.main()


if __name__ == "__main__":
    main()
