import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "paper_r113_dlr_pressure_fixture_benchmark"


def _metrics() -> dict:
    return json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r113_records_public_fixture_quantities_without_claiming_missing_ct_data():
    metrics = _metrics()
    public = metrics["direct_public_inputs"]

    assert public["length_m"] == 0.2
    assert public["start_diameter_m"] == 0.068
    assert public["exit_diameter_m"] == 0.22
    assert public["wall_thickness_range_m"] == [0.0025, 0.008]
    assert public["preload_n"] == 7000.0
    assert public["pressure_difference_pa"] == 430000.0
    assert public["rod_count"] == 12
    assert "CT-derived spatial wall-thickness map" in metrics["unavailable_inputs_not_invented"]


def test_r113_is_warning_free_mesh_resolved_and_at_the_published_strain_scale():
    metrics = _metrics()

    assert len(metrics["records"]) == 5
    assert metrics["solver"]["all_runs_warning_free"] is True
    assert metrics["mesh_check_passed"] is True
    assert metrics["mesh_profile_nrmse"] < 0.02
    assert metrics["nominal_within_published_strain_plausibility_band"] is True
    assert 0.5 < metrics["nominal_to_reported_maximum_strain_ratio"] < 1.5


def test_r113_retains_geometry_and_orientation_uncertainty_as_controls():
    metrics = _metrics()
    records = metrics["records"]

    assert metrics["published_thickness_envelope_maximum_strain"]["thin_to_thick_ratio"] > 4.0
    nominal = records["midpoint_thickness"]["maximum_absolute_surface_strain"]
    swapped = records["orientation_swapped_control"]["maximum_absolute_surface_strain"]
    assert abs(nominal - swapped) / nominal > 0.25
    assert metrics["decision"]["transfer_fixture_dimensions_to_hm7"] is False
    assert metrics["decision"]["regenerate_mechanism_identification_truth_now"] is False


def test_r113_deterministic_outputs_are_hash_locked():
    hashes = _metrics()["stable_output_sha256"]

    assert len(hashes) == 5 * 3
    for name, expected in hashes.items():
        assert _sha256(ARTIFACT / name) == expected
