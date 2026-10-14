import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "paper_r112_explicit_sliding_flange_archetype"


def _metrics() -> dict:
    return json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r112_executes_the_predeclared_connection_topology_controls():
    metrics = _metrics()
    records = metrics["records"]

    assert set(records) == {
        "sliding_mu020",
        "sliding_near_frictionless",
        "bonded_limit",
        "sliding_mu020_refined_joint",
    }
    assert metrics["solver"]["version"] == "This is Version 2.23"
    assert metrics["solver"]["all_runs_warning_free"] is True
    assert metrics["inherits_r110_qualification_without_rerun"] is True


def test_r112_resolves_the_joint_mesh_but_not_the_full_field_discrepancy():
    metrics = _metrics()
    records = metrics["records"]
    sliding = records["sliding_mu020"]
    bonded = records["bonded_limit"]

    assert metrics["joint_mesh_refinement_nrmse"] < 0.01
    assert sliding["kinematics"]["maximum_abs_radial_slip_m"] > 1.0e-4
    assert bonded["kinematics"]["maximum_abs_radial_slip_m"] < 1.0e-7
    assert sliding["comparison"]["peak_relative_error"] < 0.05
    assert bonded["external_peak_stress_pa"] > 2.0 * sliding["external_peak_stress_pa"]
    assert metrics["qualifying_cases"] == []
    assert all(
        not record["comparison"]["meets_r108_model_agreement_limits"]
        for record in records.values()
    )


def test_r112_keeps_the_archetype_out_of_the_mechanism_truth():
    metrics = _metrics()

    assert metrics["scientific_status"] == (
        "explicit_public_topology_archetype_not_hm7_geometry"
    )
    assert metrics["geometry"]["radial_extension_status"] == (
        "uncalibrated archetype dimension"
    )
    assert metrics["loads"]["snapshot_regenerated"] is False
    assert metrics["decision"]["promote_archetype_to_truth"] is False
    assert metrics["decision"]["regenerate_mechanism_identification_truth_now"] is False


def test_r112_deterministic_outputs_are_hash_locked():
    hashes = _metrics()["stable_output_sha256"]

    assert len(hashes) == 4 * 3
    assert all(name.endswith((".inp", "_results.csv")) for name in hashes)
    for name, expected in hashes.items():
        assert _sha256(ARTIFACT / name) == expected
