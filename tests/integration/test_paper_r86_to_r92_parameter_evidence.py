import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r86_exposes_that_eiv_reportability_is_not_parameter_accuracy():
    row = _load("artifacts/paper_r86_transport_coefficient_accuracy_audit/metrics.json")
    aggregate = row["aggregate_scoring_only"]

    assert aggregate["packet_count"] == 8
    assert aggregate["coefficient_count"] == 80
    assert aggregate["median_coefficient_relative_error"] >= 1.0


def test_r88_attributes_parameter_loss_primarily_to_spatial_sparsity():
    row = _load("artifacts/paper_r88_parameter_error_source_attribution/development.json")
    exact = row["results"]["exact_full_field_clean"]["summary"]
    sparse = row["results"]["four_sensor_synchronous_complete_clean"]["summary"]
    method = "independent_nonnegative_least_squares"

    assert exact[method]["median_relative_error"] < 0.05
    assert sparse[method]["median_relative_error"] > 0.80


def test_r89_rejects_pchip_as_a_parameter_rescue():
    row = _load(
        "artifacts/paper_r89_physics_smooth_spatial_reconstruction_development/"
        "development.json"
    )
    method = "independent_nonnegative_least_squares"
    linear = row["results"]["linear"]["parameter_summary"][method]
    pchip = row["results"]["pchip_meridional"]["parameter_summary"][method]

    assert row["results"]["pchip_meridional"]["structure"][
        "truth_covered_scoring_only"
    ]
    assert pchip["median_relative_error"] > linear["median_relative_error"]


def test_r92_confirms_the_frozen_seven_position_parameter_tier():
    row = _load("artifacts/paper_r92_seven_sensor_parameter_confirmation/metrics.json")
    aggregate = row["aggregate_scoring_only"]

    assert aggregate == {
        "replicate_count": 6,
        "truth_structure_coverage_count": 6,
        "unique_truth_structure_count": 3,
        "fallback_count": 3,
        "joint_adequacy_count": 6,
    }
    for record in row["records"]:
        error = record["oracle_observable_parameter_error_scoring_only"]
        assert error["median_relative_error"] <= 0.50
        assert error["maximum_relative_error"] <= 1.00


def test_r93_boundary_loads_do_not_rescue_sparse_noisy_parameters():
    row = _load("artifacts/paper_r93_boundary_aware_parameter_development/development.json")
    method = "shared_opening_ratio_profile"
    exact = row["results"]["exact_full_field_clean"][method]
    sparse = row["results"]["seven_position_asynchronous_missing_noisy"][method]

    assert abs(exact["estimated_shared_opening_modulation_ratio"] - 17.0) < 0.25
    assert exact["median_relative_error_scoring_only"] < 0.40
    assert sparse["median_relative_error_scoring_only"] > 0.99
