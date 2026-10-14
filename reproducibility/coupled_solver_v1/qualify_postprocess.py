"""New endpoint/inversion tests only, original qualifications not rerun."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if __name__=="__main__":
    sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
    from reproducibility.source_geometry_v6 import qualify_software as harness
    harness.PROTOCOL_PATH=ROOT/"reproducibility/coupled_solver_v1/postprocess-software-protocol-20261013.json"
    harness.main()
