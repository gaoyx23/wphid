import json
from pathlib import Path

from experiments import paper_h4bi_direct_forward_response as identification
from experiments import paper_h4bj_open_set_forward_adequacy as failed_open_set
from experiments import paper_h4bk_variance_budget_open_set as open_set
from experiments import paper_h4bl_variance_budget_confirmation as open_set_confirmation
from experiments import paper_h5a_identification_health_bridge as prognosis
from experiments import paper_h5b_all_family_safe_prognosis as all_family_prognosis
from experiments import paper_h5c_all_family_prognosis_confirmation as prognosis_confirmation
from experiments import paper_h5d_calibrated_health_life_intervals as initial_intervals
from experiments import paper_h5e_mission_envelope_abstention as rectangular_envelope
from experiments import paper_h5f_joint_envelope_audit as joint_envelope_audit
from experiments import paper_h5g_joint_support_guard as joint_guard
from experiments import paper_h5h_safe_interval_envelope_confirmation as failed_interval_confirmation
from experiments import paper_h5i_seed_blocked_simultaneous_intervals as blocked_intervals
from experiments import paper_h5j_safe_prognosis_interval_confirmation as interval_confirmation


def _payload(module):
    return json.loads(Path(module.DEFAULT_OUTPUT).read_text(encoding="utf-8"))


def test_direct_forward_development_recovers_all_supported_families_safely():
    payload = _payload(identification)
    aggregate = payload["aggregate"]
    assert payload["protocol_sha256"] == identification.PROTOCOL_HASH
    assert payload["status"] == "DEVELOPMENT"
    assert aggregate["decision"] == "DEVELOPMENT-GO"
    assert aggregate["positive_correct_outputs"] == 48
    assert aggregate["positive_wrong_outputs"] == 0
    assert aggregate["exact_equivalence_abstentions"] == 12
    assert aggregate["exact_equivalence_count"] == 12


def test_health_bridge_uses_matching_identification_artifact_and_passes_each_case():
    payload = _payload(prognosis)
    aggregate = payload["aggregate"]
    assert payload["protocol_sha256"] == prognosis.PROTOCOL_HASH
    assert (
        payload["protocol"]["identification_protocol_sha256"]
        == identification.PROTOCOL_HASH
    )
    assert payload["status"] == "DEVELOPMENT"
    assert aggregate["identification_status"].endswith("not confirmation")
    assert aggregate["decision"] == "DEVELOPMENT-GO"
    assert aggregate["case_count"] == 24
    assert aggregate["case_pass_rate"] == 1.0
    assert aggregate["maximum_rul_relative_error"] <= 0.10
    assert aggregate["maximum_damage_absolute_error"] <= 0.04
    assert len(payload["cases"]) == 6
    assert all(row["pass_rate"] == 1.0 for row in payload["cases"])


def test_health_bridge_outputs_health_and_life_for_every_supported_program():
    payload = _payload(prognosis)
    expected_pairs = {
        (family, program)
        for family in prognosis.PROTOCOL["families"]
        for program in prognosis.PROTOCOL["future_programs"]
    }
    actual_pairs = {(row["family"], row["program"]) for row in payload["records"]}
    assert actual_pairs == expected_pairs
    assert all(row["truth_rul_cycles"] > 0 for row in payload["records"])
    assert all(len(row["predicted_rul_cycles"]) == 4 for row in payload["records"])
    assert all(
        set(row["health_index_absolute_error_by_horizon"]) == {"10", "20"}
        for row in payload["records"]
    )


def test_variance_budget_open_set_development_retains_failure_and_has_margin():
    failed = _payload(failed_open_set)
    payload = _payload(open_set)
    aggregate = payload["aggregate"]
    assert failed["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"
    assert failed["aggregate"]["candidate_out_false_admissions"] == 2
    assert payload["protocol_sha256"] == open_set.PROTOCOL_HASH
    assert aggregate["decision"] == "DEVELOPMENT-GO"
    assert aggregate["positive_correct_outputs"] == 48
    assert aggregate["positive_wrong_outputs"] == 0
    assert aggregate["exact_equivalence_abstentions"] == 12
    assert aggregate["candidate_out_abstentions"] == 24
    assert aggregate["candidate_out_false_admissions"] == 0
    threshold = payload["protocol"]["maximum_cv_rmse_to_repeat_sd"]
    positive = [row for row in payload["records"] if row["expected_family"] != "abstain"]
    candidate_out = [row for row in payload["records"] if row["candidate_out"]]
    assert max(row["forward_cv_rmse_to_repeat_sd"] for row in positive) < threshold
    assert min(row["forward_cv_rmse_to_repeat_sd"] for row in candidate_out) > threshold


def test_locked_open_set_confirmation_is_disjoint_and_passes_with_margin():
    development = _payload(open_set)
    confirmation = _payload(open_set_confirmation)
    aggregate = confirmation["aggregate"]
    assert set(open_set_confirmation.SEEDS).isdisjoint(
        open_set.PROTOCOL["development_seeds"]
    )
    assert confirmation["protocol_sha256"] == open_set_confirmation.PROTOCOL_HASH
    assert confirmation["status"] == "CONFIRMATION-LOCKED"
    assert aggregate["decision"] == "CONFIRMATION-GO"
    assert aggregate["positive_correct_outputs"] == 60
    assert aggregate["positive_wrong_outputs"] == 0
    assert aggregate["exact_equivalence_abstentions"] == 15
    assert aggregate["candidate_out_abstentions"] == 30
    assert aggregate["candidate_out_false_admissions"] == 0
    threshold = confirmation["protocol"]["maximum_cv_rmse_to_repeat_sd"]
    positive = [
        row for row in confirmation["records"]
        if row["expected_family"] != "abstain"
    ]
    candidate_out = [row for row in confirmation["records"] if row["candidate_out"]]
    assert max(row["forward_cv_rmse_to_repeat_sd"] for row in positive) < threshold
    assert min(row["forward_cv_rmse_to_repeat_sd"] for row in candidate_out) > threshold
    assert development["aggregate"]["decision"] == "DEVELOPMENT-GO"


def test_safe_all_family_prognosis_development_covers_every_family_and_program():
    payload = _payload(all_family_prognosis)
    aggregate = payload["aggregate"]
    assert payload["protocol_sha256"] == all_family_prognosis.PROTOCOL_HASH
    assert aggregate["decision"] == "DEVELOPMENT-GO"
    assert aggregate["identification_authorized"] == 16
    assert aggregate["identification_count"] == 16
    assert aggregate["forecast_case_count"] == 48
    assert aggregate["forecast_case_pass_rate"] == 1.0
    assert aggregate["maximum_rul_relative_error"] <= 0.10
    assert aggregate["maximum_damage_absolute_error"] <= 0.04
    assert aggregate["real_data_status"] == "not evaluated"
    assert len(payload["cases"]) == 12
    assert all(row["pass_rate"] == 1.0 for row in payload["cases"])
    safety = payload["safety_interface"]
    assert safety["candidate_out_abstentions"] == safety["candidate_out_count"] == 30
    assert safety["candidate_out_prognosis_outputs"] == 0
    assert safety["equivalence_abstentions"] == safety["equivalence_count"] == 15
    assert safety["equivalence_prognosis_outputs"] == 0


def test_locked_all_family_prognosis_confirmation_is_independent_and_safe():
    payload = _payload(prognosis_confirmation)
    aggregate = payload["aggregate"]
    assert set(prognosis_confirmation.IDENTIFICATION_SEEDS).isdisjoint(
        all_family_prognosis.PROTOCOL["identification_seeds"]
    )
    assert set(prognosis_confirmation.HEALTH_OBSERVATION_SEEDS).isdisjoint(
        all_family_prognosis.PROTOCOL["health_observation_seeds"]
    )
    assert payload["protocol_sha256"] == prognosis_confirmation.PROTOCOL_HASH
    assert payload["status"] == "CONFIRMATION-LOCKED"
    assert aggregate["decision"] == "CONFIRMATION-GO"
    assert aggregate["identification_authorized"] == 20
    assert aggregate["identification_count"] == 20
    assert aggregate["forecast_case_count"] == 60
    assert aggregate["forecast_case_pass_rate"] == 1.0
    assert aggregate["maximum_rul_relative_error"] <= 0.10
    assert aggregate["maximum_damage_absolute_error"] <= 0.04
    assert aggregate["overall_passive_h4_decision"] == "NO-GO"
    assert aggregate["real_data_status"] == "not evaluated"
    assert len(payload["cases"]) == 12
    assert all(row["pass_rate"] == 1.0 for row in payload["cases"])
    safety = payload["safety_interface"]
    assert safety["candidate_out_prognosis_outputs"] == 0
    assert safety["equivalence_prognosis_outputs"] == 0


def test_interval_failure_is_retained_and_seed_blocking_repairs_independence():
    initial = _payload(initial_intervals)
    failed = _payload(failed_interval_confirmation)
    repaired = _payload(blocked_intervals)
    assert initial["aggregate"]["decision"] == "DEVELOPMENT-GO"
    assert failed["aggregate"]["decision"] == "CONFIRMATION-NO-GO"
    assert failed["interval_summary"]["aggregate_damage_coverage"]["10"] < 0.90
    assert failed["guard_summary"]["gate_pass"]
    assert repaired["protocol_sha256"] == blocked_intervals.PROTOCOL_HASH
    assert repaired["aggregate"]["decision"] == "DEVELOPMENT-GO"
    assert repaired["aggregate"]["h5h_decision_retained"] == "CONFIRMATION-NO-GO"
    assert all(
        row["calibration_unit_count"] == 20
        and row["rul_rank_one_based"] == 19
        for row in repaired["finite_sample_ranks"].values()
    )


def test_joint_guard_retains_rectangular_failure_and_blocks_unsafe_outputs():
    rectangular = _payload(rectangular_envelope)
    failed = _payload(joint_envelope_audit)
    repaired = _payload(joint_guard)
    assert rectangular["aggregate"]["decision"] == "DEVELOPMENT-GO"
    assert failed["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"
    assert failed["aggregate"]["joint_candidate_false_admissions"] == 6
    assert repaired["protocol_sha256"] == joint_guard.PROTOCOL_HASH
    assert repaired["aggregate"]["decision"] == "DEVELOPMENT-GO"
    assert repaired["aggregate"]["validated_admission_rate"] == 1.0
    assert repaired["aggregate"]["near_support_admission_rate"] == 1.0
    assert repaired["aggregate"]["invalid_abstention_rate"] == 1.0
    assert repaired["aggregate"]["unsafe_prognosis_outputs"] == 0


def test_final_safe_interval_confirmation_is_independent_narrow_and_calibrated():
    payload = _payload(interval_confirmation)
    aggregate = payload["aggregate"]
    assert set(interval_confirmation.EVALUATION_SEEDS).isdisjoint(
        blocked_intervals.CALIBRATION_SEEDS
    )
    assert set(interval_confirmation.EVALUATION_SEEDS).isdisjoint(
        blocked_intervals.EVALUATION_SEEDS
    )
    assert payload["protocol_sha256"] == interval_confirmation.PROTOCOL_HASH
    assert payload["status"] == "CONFIRMATION-LOCKED"
    assert aggregate["decision"] == "CONFIRMATION-GO"
    assert aggregate["evaluation_case_count"] == 360
    assert aggregate["aggregate_rul_coverage"] >= 0.90
    assert min(aggregate["aggregate_damage_coverage"].values()) >= 0.90
    assert aggregate["maximum_relative_rul_interval_width"] <= 0.20
    assert aggregate["maximum_normalized_damage_interval_width"] <= 0.10
    assert aggregate["joint_guard_confirmation_pass"]
    assert aggregate["unsafe_prognosis_outputs_in_guard_confirmation"] == 0
    assert aggregate["h5h_decision_retained"] == "CONFIRMATION-NO-GO"
    assert len(payload["cases"]) == 12
    assert all(row["pass"] for row in payload["cases"])
