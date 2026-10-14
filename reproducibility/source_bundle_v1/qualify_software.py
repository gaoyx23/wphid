"""Reuse the frozen bounded qualification harness with an independent protocol."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    import wphid.source_bundle_v1.bundle as bundle
    if not Path(bundle.__file__).resolve().is_relative_to(ROOT / "src"):
        raise SystemExit("unexpected source bundle import")
    from reproducibility.source_geometry_v6 import qualify_software as harness
    # Runtime configuration only: the frozen harness file and its original
    # protocol/results remain unchanged; pytest runs only this new test file.
    harness.PROTOCOL_PATH = ROOT / "reproducibility/source_bundle_v1/software-protocol.json"
    print("imported_source_bundle:", str(Path(bundle.__file__).resolve()))
    harness.main()


if __name__ == "__main__":
    main()
