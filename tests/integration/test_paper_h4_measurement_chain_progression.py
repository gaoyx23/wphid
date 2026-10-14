import json
from pathlib import Path

from experiments import h4ar_independent_finite_volume_response_source as source
from experiments import paper_h4au_measurement_chain_audit as nominal_audit
from experiments import paper_h4bb_measurement_chain_confirmation as mean_confirmation
from experiments import paper_h4bd_practical_adequacy_extended_panel as extended_mean
from experiments import paper_h4be_robust_spatial_consensus as robust_development
from experiments import paper_h4bf_robust_measurement_chain_confirmation as robust_confirmation
from experiments import paper_h4bg_expanded_family_recovery as expanded_recovery
from experiments import paper_h4bh_factorial_baseline_design as factorial_design


def _payload(module):
    return json.loads(Path(module.DEFAULT_OUTPUT).read_text(encoding="utf-8"))


def test_measurement_packet_exposes_blanks_realized_doses_and_replicates():
    packet = source.generate_packet(
        45501,
        "oxygen_damage_power",
        0.9,
        relative_intervention_levels=(0.10, 0.20, 0.30),
        include_measurement_metadata=True,
        readout_replicates=3,
        dose_reporting="sensor_weighted",
        block_count=4,
    )
    assert len(packet["rows"]) == 4 * 2 * 3 * 2 * 3
    assert len(packet["blank_responses"]) == 4
    assert packet["readout_replicates"] == 3
    assert packet["dose_reporting"] == "sensor_weighted"
    assert all(
        "realized_log_oxygen_change" in row
        and "realized_log_damage_change" in row
        for row in packet["rows"]
    )


def test_retained_measurement_chain_failures_are_not_overwritten():
    nominal = _payload(nominal_audit)
    mean_locked = _payload(mean_confirmation)
    extended = _payload(extended_mean)
    assert nominal["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"
    assert nominal["aggregate"]["in_grammar_wrong_outputs"] > 0
    assert mean_locked["aggregate"]["decision"] == "CONFIRMATION-NO-GO"
    assert mean_locked["aggregate"]["in_grammar_correct_outputs"] == 52
    assert extended["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"


def test_robust_confirmation_seeds_are_disjoint_from_all_method_development():
    confirmation_seeds = set(robust_confirmation.SEEDS)
    assert confirmation_seeds.isdisjoint(robust_development.PROTOCOL["seeds"])
    assert confirmation_seeds.isdisjoint(mean_confirmation.SEEDS)
    assert confirmation_seeds.isdisjoint(extended_mean.PROTOCOL["seeds"])


def test_retained_robust_confirmation_matches_locked_protocol_and_safety_gates():
    development = _payload(robust_development)
    confirmation = _payload(robust_confirmation)
    assert development["protocol_sha256"] == robust_development.PROTOCOL_HASH
    assert development["aggregate"]["decision"] == "DEVELOPMENT-GO"
    assert confirmation["protocol_sha256"] == robust_confirmation.PROTOCOL_HASH
    assert confirmation["status"] == "CONFIRMATION-LOCKED"
    assert confirmation["aggregate"]["decision"] == "CONFIRMATION-GO"
    assert confirmation["aggregate"]["in_grammar_correct_outputs"] == 60
    assert confirmation["aggregate"]["in_grammar_wrong_outputs"] == 0
    assert confirmation["aggregate"]["equivalence_abstentions"] == 30
    assert confirmation["aggregate"]["candidate_out_abstentions"] == 60
    assert confirmation["aggregate"]["candidate_out_false_admissions"] == 0
    assert confirmation["aggregate"]["overall_passive_h4_decision"] == "NO-GO"
    assert all(
        row["in_grammar_correct_outputs"] == row["in_grammar_count"] == 20
        and row["in_grammar_wrong_outputs"] == 0
        and row["equivalence_abstentions"] == row["equivalence_count"] == 10
        and row["candidate_out_abstentions"] == row["candidate_out_count"] == 20
        for row in confirmation["scenario_summary"]
    )


def test_expanded_family_failures_are_retained_without_running_confirmation():
    heterogeneous = _payload(expanded_recovery)
    factorial = _payload(factorial_design)
    assert heterogeneous["protocol_sha256"] == expanded_recovery.PROTOCOL_HASH
    assert heterogeneous["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"
    assert heterogeneous["aggregate"]["positive_wrong_outputs"] == 2
    assert factorial["protocol_sha256"] == factorial_design.PROTOCOL_HASH
    assert factorial["status"] == "DEVELOPMENT-EARLY-STOPPED"
    assert factorial["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"
    assert factorial["aggregate"]["positive_wrong_outputs"] == 3
    assert factorial["aggregate"]["unrun_planned_seeds"] == [45602, 45603, 45604]
