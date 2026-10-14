import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_domain_shift_development_failures_are_preserved_before_final_rule():
    qualification = load(
        "artifacts/paper_r100_geometry_material_defect_transfer/"
        "qualification_diagnostic.json"
    )
    r100 = load(
        "artifacts/paper_r100_geometry_material_defect_transfer/development.json"
    )
    r102 = load(
        "artifacts/paper_r102_domain_shift_safe_recovery_development/development.json"
    )
    r103 = load(
        "artifacts/paper_r103_strict_estimator_agreement_redecision/metrics.json"
    )
    r104 = load(
        "artifacts/paper_r104_domain_shift_safe_recovery_confirmation/metrics.json"
    )

    assert qualification["recovery_runs_completed_before_diagnostic"] == 0
    assert qualification["correction_before_any_recovery"][
        "structure_or_parameter_thresholds_changed"
    ] is False
    assert r100["development_gate"] is False
    assert r102["development_gate_all_five_cases_cover_truth"] is False
    assert r103["development_gate_all_five_cases_cover_truth"] is True
    assert r104["seed_overlap_with_development"] is False
    assert r104["combined"]["all_five_truth_coverage_count"] == 13
    assert r104["combined"]["replicate_count"] == 15
    assert r104[
        "confirmation_gate_all_records_cover_five_truth_mechanisms"
    ] is False


def test_frozen_cross_scale_union_covers_held_out_domains_without_false_uniqueness():
    r105 = load(
        "artifacts/paper_r105_cross_scale_union_domain_shift_development/"
        "development.json"
    )
    r106 = load(
        "artifacts/paper_r106_cross_scale_union_domain_shift_confirmation/metrics.json"
    )

    assert r105["development_record_count"] == 8
    assert r105["all_five_truth_coverage_count"] == 8
    assert r105["development_gate"] is True
    assert r105["confirmation_authorized"] is True

    combined = r106["combined"]
    assert r106["seed_overlap_with_all_prior_domain_shift_panels"] is False
    assert combined["replicate_count"] == 20
    assert combined["all_five_truth_coverage_count"] == 20
    assert combined["carbon_exact_count"] == 20
    assert combined["crack_truth_coverage_count"] == 20
    assert combined["transport_truth_coverage_count"] == 20
    assert combined["crack_unique_truth_count"] == 0
    assert combined["transport_unique_truth_count"] == 0
    assert r106[
        "confirmation_gate_all_records_cover_five_truth_mechanisms"
    ] is True


def test_fullfield_data_contracts_the_domain_shift_identified_sets():
    sparse = load(
        "artifacts/paper_r106_cross_scale_union_domain_shift_confirmation/metrics.json"
    )["combined"]
    full = load(
        "artifacts/paper_r107_fullfield_domain_shift_completion_audit/metrics.json"
    )["aggregate"]

    assert full["all_five_truth_coverage_count"] == 5
    assert full["complete_exact_local_support_count"] == 5
    assert full["transport_minimum_error_truth_count"] == 5
    assert full["transport_truth_coverage_count"] == 5
    assert full["transport_unique_truth_count"] == 3
    assert full["mean_transport_candidate_count"] < sparse[
        "mean_transport_family_count"
    ]
