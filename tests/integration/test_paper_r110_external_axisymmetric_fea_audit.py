import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "paper_r110_external_axisymmetric_fea_audit"


def _metrics() -> dict:
    return json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r110_external_solver_is_qualified_before_nozzle_comparison():
    metrics = _metrics()
    solver = metrics["solver"]
    qualification = metrics["qualification"]

    assert solver["version_output"] == "This is Version 2.23"
    assert solver["all_runs_warning_free"] is True
    assert solver["all_runs_completed"] is True
    assert qualification["all_checks_pass"] is True
    assert all(qualification["checks"].values())
    assert metrics["nozzle_runs_authorized_after_qualification"] is True


def test_r110_full_bell_mesh_is_converged_under_frozen_limits():
    convergence = _metrics()["nozzle_mesh"]["convergence"]
    limits = convergence["predeclared_limits"]

    assert convergence["passed"] is True
    assert convergence["von_mises_nrmse"] <= limits["maximum_von_mises_nrmse"]
    assert convergence["peak_relative_error"] <= limits[
        "maximum_peak_relative_error"
    ]
    assert convergence["spatial_correlation"] >= limits[
        "minimum_spatial_correlation"
    ]


def test_r110_records_the_external_model_form_discrepancy_without_truth_rewrite():
    metrics = _metrics()

    assert metrics["scientific_status"] == (
        "external_model_form_discrepancy_requires_resolution"
    )
    assert set(metrics["boundary_cases"]) == {
        "free_anchor",
        "inlet_axial_roller",
        "inlet_clamped",
    }
    assert metrics["boundary_variants_meeting_all_r108_limits"] == []
    assert metrics["any_boundary_variant_meets_all_r108_limits"] is False
    assert metrics["decision"][
        "current_reduced_stress_closure_externally_supported"
    ] is False
    assert metrics["decision"]["regenerate_mechanism_identification_truth_now"] is False
    assert metrics["source"]["trajectory_regenerated"] is False


def test_r110_hash_locks_only_deterministic_decks_and_normalized_results():
    hashes = _metrics()["output_sha256"]

    assert hashes
    assert all(name.endswith((".inp", "_results.csv")) for name in hashes)
    assert not any(name.endswith((".frd", ".sta", ".cvg")) for name in hashes)
    for name, expected in hashes.items():
        assert _sha256(ARTIFACT / name) == expected
