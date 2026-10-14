"""Inspect completion and CPU records only, never scientific parameter scores."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('original_progress', ROOT / 'reproducibility/comparison_resume_v2/progress.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
p.OUTPUT = ROOT / 'artifacts/paper_comparison_resume_v4/gaussian_fresh_confirmation'

if __name__ == '__main__':
    p.main()
