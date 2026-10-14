"""Run only the new conservative balance checks through the frozen harness."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    import wphid.coupled_balance_v1.assembler as assembler
    if not Path(assembler.__file__).resolve().is_relative_to(ROOT / "src"):
        raise SystemExit("unexpected coupled balance import")
    from reproducibility.source_geometry_v6 import qualify_software as harness
    harness.PROTOCOL_PATH = ROOT / "reproducibility/coupled_balance_v1/software-protocol.json"
    print("imported_coupled_balance:", str(Path(assembler.__file__).resolve()))
    harness.main()


if __name__ == "__main__":
    main()
