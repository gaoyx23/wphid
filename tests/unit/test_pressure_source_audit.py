"""Saved-artifact bookkeeping checks; no simulator or noise generation."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "artifacts/_audits/pressure_source_dependence/audit.json"


@pytest.fixture
def report():
    if not AUDIT.exists(): pytest.skip("Local diagnostic artifact unavailable")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_completed_audit_has_only_missing_sources(report):
    assert report["complete"]
    assert set(report["cases"]) == {"Darcy__directional", "Forchheimer__isotropic", "Forchheimer__directional"}
    assert sum(len(v) for v in report["cases"].values()) == 12
    assert report["new_forward_runs"] == report["new_noise_draws"] == 0
    assert not report["common_modes_included"]


def test_saved_covariances_are_positive_semidefinite(report):
    for cases in report["cases"].values():
        for result in cases.values():
            cov = np.asarray(result["independent_pressure_covariance"])
            assert cov.shape == (18, 18)
            np.testing.assert_allclose(cov, cov.T)
            assert np.linalg.eigvalsh(cov).min() >= -1.e-12*np.linalg.norm(cov)


def test_paired_covariance_bound_and_trace_ratio(report):
    for source, pairs in report["pairs"].items():
        assert len(pairs) == 6
        for key, pair in pairs.items():
            a, b = key.split("__versus__")
            cov = np.asarray(pair["paired_pressure_covariance"])
            independent = sum(np.asarray(report["cases"][source][name]["independent_pressure_covariance"]) for name in (a, b))
            assert pair["paired_to_independent_trace_ratio"] == pytest.approx(np.trace(cov)/np.trace(independent))
            assert 0. <= pair["paired_to_independent_trace_ratio"] <= 2.+1.e-12


def test_indicator_is_saved_only_for_nonlinear_candidates(report):
    for cases in report["cases"].values():
        for name, case in cases.items():
            assert bool(case["face_nonlinearity_indicators"]) == name.startswith("Forchheimer")
