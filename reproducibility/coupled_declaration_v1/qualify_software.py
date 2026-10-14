"""Only new declared-candidate or censored-law checks, with frozen receipts."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kind", required=True, choices=("candidates", "observation"))
    args, remaining = parser.parse_known_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    filename = "physical-candidates-protocol-20261013.json" if args.kind == "candidates" else "censored-observation-protocol-20261013.json"
    protocol_path = ROOT / "reproducibility/coupled_declaration_v1" / filename
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    from reproducibility.coupled_observation_v1.qualify_software import verify_dependencies
    verify_dependencies(json.loads(protocol_path.read_text(encoding="utf-8")))
    from reproducibility.source_geometry_v6 import qualify_software as harness
    sys.argv = [sys.argv[0], *remaining]
    harness.PROTOCOL_PATH = protocol_path
    harness.main()


if __name__ == "__main__":
    main()
