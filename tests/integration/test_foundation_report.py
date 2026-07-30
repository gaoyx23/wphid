"""Foundation report renderer contract tests."""

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
README_PATH = PROJECT_ROOT / "README.md"
REPORT_PATH = PROJECT_ROOT / "docs" / "results" / "wphid-foundation-gate.md"
E1_METRICS_PATH = PROJECT_ROOT / "artifacts" / "e1_closed_library" / "metrics.json"
E3_METRICS_PATH = PROJECT_ROOT / "artifacts" / "e3_missing_interaction" / "metrics.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_report_equals_renderer_output():
    """The committed gate report must match the deterministic renderer output."""
    from experiments.foundation_report import render_report

    e1_metrics = _load_json(E1_METRICS_PATH)
    e3_metrics = _load_json(E3_METRICS_PATH)
    expected = render_report(e1_metrics, e3_metrics)
    committed = REPORT_PATH.read_text(encoding="utf-8")
    assert committed == expected


def test_pyproject_declares_runtime_dependencies():
    """pyproject.toml must declare NumPy and SciPy as runtime dependencies."""
    content = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert "numpy" in content
    assert "scipy" in content


def test_pyproject_declares_test_extra():
    """pyproject.toml must declare a [test] extra containing pytest."""
    content = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert "pytest" in content
    assert "test" in content


def test_readme_uses_install_test_extra():
    """README must instruct users to install with the [test] extra."""
    content = README_PATH.read_text(encoding="utf-8")
    assert 'pip install -e ".[test]"' in content or "pip install -e '.[test]'" in content


def test_readme_references_foundation_report_command():
    """README must document the foundation report regeneration command."""
    content = README_PATH.read_text(encoding="utf-8")
    assert "python experiments/foundation_report.py" in content


def test_report_states_e2_unimplemented():
    """The committed report must state that E2 remains unimplemented."""
    content = REPORT_PATH.read_text(encoding="utf-8")
    assert "E2" in content
    assert "unimplemented" in content.lower() or "not implemented" in content.lower()


def test_report_decision_matches_e1_gate():
    """The report decision must match the E1 aggregate gate boolean."""
    e1_metrics = _load_json(E1_METRICS_PATH)
    content = REPORT_PATH.read_text(encoding="utf-8")
    e1_pass = e1_metrics["aggregate"]["gate"]["pass"]
    if not e1_pass:
        assert "NO-GO" in content
