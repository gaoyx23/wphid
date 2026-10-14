import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "paper_r111_orthotropic_compliant_attachment_sensitivity"


def _metrics() -> dict:
    return json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r111_runs_the_complete_predeclared_factorial_panel():
    metrics = _metrics()
    panel = metrics["predeclared_panel"]

    assert panel["through_thickness_modulus_ratios"] == [0.25, 0.5, 1.0]
    assert panel["radial_cte_ratios"] == [0.5, 1.0, 2.0]
    assert panel["interface_stiffness_ratios"] == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert panel["case_count"] == 45
    assert len(metrics["records"]) == 45
    assert metrics["run_summary"]["all_completed"] is True
    assert metrics["run_summary"]["all_warning_free"] is True


def test_r111_retains_the_discrepancy_and_does_not_select_a_truth_model():
    metrics = _metrics()
    best = metrics["best_diagnostic_case"]

    assert metrics["scientific_status"] == (
        "bounded_panel_does_not_resolve_r110_discrepancy"
    )
    assert metrics["qualifying_case_count"] == 0
    assert metrics["qualifying_cases"] == []
    assert best["comparison"]["meets_r108_model_agreement_limits"] is False
    assert best["comparison"]["nrmse_by_external_range"] > 0.2
    assert best["comparison"]["peak_relative_error"] > 0.2
    assert metrics["decision"]["select_best_case_as_truth"] is False
    assert metrics["decision"]["regenerate_mechanism_identification_truth_now"] is False


def test_r111_deterministic_outputs_are_hash_locked():
    hashes = _metrics()["stable_output_sha256"]

    assert len(hashes) == 45 * 3
    assert all(name.endswith((".inp", "_results.csv")) for name in hashes)
    for name, expected in hashes.items():
        assert _sha256(ARTIFACT / name) == expected
