import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "paper_r108_axisymmetric_thermoelastic_audit"


def test_r108_solver_is_qualified_before_model_form_comparison():
    metrics = json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))
    qualification = metrics["solver_qualification"]
    limits = qualification["predeclared_limits"]

    assert qualification["all_checks_pass"] is True
    assert qualification["lame_radial_relative_l2_error"] < limits[
        "maximum_lame_relative_l2_error"
    ]
    assert qualification["lame_hoop_relative_l2_error"] < limits[
        "maximum_lame_relative_l2_error"
    ]
    assert qualification["uniform_free_heating_maximum_von_mises_pa"] < limits[
        "maximum_uniform_free_heating_stress_pa"
    ]
    assert qualification["reference_hottest_station_64_to_128_nrmse"] < limits[
        "maximum_64_to_128_mission_field_nrmse"
    ]


def test_r108_reduced_stress_agrees_with_reference_in_all_five_domains():
    metrics = json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))
    aggregate = metrics["aggregate"]
    limits = metrics["predeclared_model_agreement_limits"]

    assert len(metrics["records"]) == 5
    assert aggregate["case_count"] == 5
    assert aggregate["case_pass_count"] == 5
    assert aggregate["all_cases_meet_model_agreement_limits"] is True
    assert metrics["audit_supports_current_reduced_stress_closure"] is True
    assert aggregate["nrmse_by_reference_range"] <= limits[
        "nrmse_by_reference_range"
    ]
    assert aggregate["peak_relative_error"] <= limits["peak_relative_error"]
    assert aggregate["spatial_correlation"] >= limits[
        "minimum_spatial_correlation"
    ]
    assert aggregate["crack_threshold_classification_agreement"] >= limits[
        "minimum_crack_threshold_classification_agreement"
    ]


def test_r108_persisted_comparison_fields_are_complete_and_finite():
    with np.load(ARTIFACT / "comparison_fields.npz", allow_pickle=False) as fields:
        for case in (
            "reference_bartz",
            "geometry_envelope_shift",
            "material_batch_shift",
            "initial_defect_shift",
            "combined_shift",
        ):
            reference = fields[f"{case}__reference"]
            reduced = fields[f"{case}__reduced"]
            assert reference.shape == reduced.shape == (31, 8)
            assert np.all(np.isfinite(reference))
            assert np.all(np.isfinite(reduced))
