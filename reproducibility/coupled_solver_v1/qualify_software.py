"""Run only the new complete-solver software checks in an exclusive output."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if __name__=="__main__":
    if Path.cwd().resolve()!=ROOT:raise SystemExit("run from worktree")
    sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
    from reproducibility.coupled_observation_v1.qualify_software import verify_dependencies
    from reproducibility.source_geometry_v6 import qualify_software as harness
    protocol=ROOT/"reproducibility/coupled_solver_v1/software-protocol-20261013.json"
    verify_dependencies(json.loads(protocol.read_text(encoding="utf-8")))
    harness.PROTOCOL_PATH=protocol
    harness.main()
