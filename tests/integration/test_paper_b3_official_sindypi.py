import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_official_sindypi_runs_in_isolated_legacy_environment():
    root = Path(__file__).resolve().parents[2]
    if not (root / "tmp" / "pysindy175").exists():
        pytest.skip(
            "optional PySINDy 1.7.5 environment is rebuildable from the retained wheel"
        )
    completed = subprocess.run(
        [sys.executable, "experiments/paper_b3_official_sindypi.py", "--smoke"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "optimizer": "SINDyPI",
        "pysindy": "1.7.5",
        "feature_count": 19,
        "equation_count": 3,
    }
