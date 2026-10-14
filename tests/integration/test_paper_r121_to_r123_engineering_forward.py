import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r121_realistic_boundary_composition_does_not_resolve_directional_truth():
    payload = _load(
        "artifacts/paper_r121_candidate_forward_engineering_observability/development.json"
    )
    assert payload["confirmation_seeds_used"] is False
    assert payload["full_field_candidate_archives_created"] is False
    assert payload["new_candidate_forward_runs"] == 24
    assert payload["instrumented_boundary_information_sufficient"] is False
    for tier in ("exit_spectroscopy_only", "instrumented_boundary_composition"):
        directional = payload["aggregate"][tier]["strong_directional"]
        assert directional["truth_coverage_count"] == 0
        assert directional["wrong_only_count"] == 3


def test_r122_attributes_r121_failure_to_signal_loss_before_regression():
    payload = _load(
        "artifacts/paper_r122_engineering_signal_attribution/diagnostic.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["confirmation_seeds_used"] is False
    summary = payload["summary"]
    assert summary["engineering_boundary_signal_below_one_noise_rms"] is True
    assert summary["maximum_engineering_block_rms_signal_to_noise"] < 0.02
    assert summary["maximum_midwall_rms_signal_to_noise"] < 1.0


def test_r123_full_coupled_truth_is_stable_and_propagates_transport_feedback():
    payload = _load(
        "artifacts/paper_r123_full_coupled_transport_truth/metrics.json"
    )
    assert payload["role"] == "new_truth_source_qualification_not_identification"
    assert payload["new_full_coupled_runs"] == 8
    assert payload["all_runs_finite"] is True
    preflight = payload["stability_preflight"]
    assert preflight["same_time_step_for_both_topologies"] is True
    assert preflight["selected_common_time_step_seconds"] == 0.01
    assert preflight["rejected_directional_explicit_number"] > 0.5
    assert payload["summary"]["transport_feedback_reaches_carbon_and_crack"] is True


def test_r123_engineering_signal_remains_below_declared_noise():
    payload = _load(
        "artifacts/paper_r123_full_coupled_transport_truth/metrics.json"
    )
    summary = payload["summary"]
    assert summary["at_least_one_engineering_block_exceeds_one_noise_rms"] is False
    assert 0.25 < summary["maximum_engineering_block_rms_signal_to_noise"] < 0.40
    instrumented = payload["engineering_observation_separation"][
        "instrumented_research_rig"
    ]
    assert max(
        instrumented,
        key=lambda block: instrumented[block]["rms_signal_to_declared_noise"],
    ) == "inner_surface_microports_pProducts"


def test_r124_freezes_a_passive_throat_focused_observation_requirement():
    payload = _load(
        "artifacts/paper_r124_passive_observation_requirement/diagnostic.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["active_interventions_used"] is False
    assert payload["confirmation_seeds_used"] is False
    summary = payload["summary"]
    assert summary["any_existing_microport_block_rms_above_one"] is False
    assert 0.65 < summary[
        "best_all_condition_focused_location_rms_signal_to_noise"
    ] < 0.75
    assert 1.4 < summary[
        "uniform_noise_reduction_factor_for_best_all_condition_focused_rms_one"
    ] < 1.5
    assert summary[
        "best_single_condition_focused_location_exceeds_one_noise_rms"
    ] is True
    assert summary["either_difference_operator_improves_over_raw"] is False
    focused = payload["best_natural_condition_focused_location"]
    assert focused["block"] == "inner_surface_microports_pProducts"
    assert focused["condition_id"] == "high_steam_900s"
    assert focused["uniform_microport_index"] == 0


def test_r125_nominal_condition_holdout_stops_before_noise_confirmation():
    payload = _load(
        "artifacts/paper_r125_heldout_nominal_focused_observation/preflight.json"
    )
    assert payload["layout_frozen_from"] == "R124"
    assert payload["condition_excluded_from_r123_r124_signal_selection"] is True
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["noise_seeds_used"] is False
    assert payload["active_interventions_used"] is False
    assert 0.80 < payload["observed_rms_signal_to_declared_noise"] < 0.85
    assert payload["maximum_absolute_signal_to_declared_noise"] < 1.0
    assert payload["confirmation_authorized"] is False


def test_r126_focused_observation_transfer_is_material_limited():
    payload = _load(
        "artifacts/paper_r126_focused_observation_transfer/development.json"
    )
    assert payload["layout_and_condition_frozen_from"] == "R124"
    assert payload["variation_envelope_frozen_from"] == "R100"
    assert payload["active_interventions_used"] is False
    assert payload["expected_full_coupled_runs"] == 8
    assert payload["complete_cached_or_generated_runs"] == 8
    assert payload["all_runs_finite"] is True
    summary = payload["summary"]
    assert summary["cases_meeting_frozen_observation_threshold"] == 3
    assert summary["all_variation_cases_meet_frozen_observation_threshold"] is False
    assert summary["noise_confirmation_authorized"] is False
    results = payload["results"]
    assert results["material_batch_shift"][
        "meets_frozen_observation_threshold"
    ] is False
    assert 0.85 < results["material_batch_shift"][
        "rms_signal_to_declared_noise"
    ] < 0.90
    assert 1.10 < results["material_batch_shift"][
        "precision_improvement_factor_for_rms_one"
    ] < 1.20
    for case_id in (
        "geometry_envelope_shift",
        "initial_defect_shift",
        "combined_shift",
    ):
        assert results[case_id]["meets_frozen_observation_threshold"] is True


def test_r127_attributes_material_attenuation_to_through_conductivity():
    payload = _load(
        "artifacts/paper_r127_material_driver_attribution/development.json"
    )
    assert payload["observation_and_mission_frozen_from"] == "R124"
    assert payload["material_endpoints_frozen_from"] == "R100"
    assert payload["active_interventions_used"] is False
    assert payload["new_endpoint_simulations_avoided"] == 4
    assert payload["expected_new_one_factor_runs"] == 6
    assert payload["complete_cached_or_generated_one_factor_runs"] == 6
    assert payload["all_runs_finite"] is True
    summary = payload["summary"]
    assert summary["primary_attenuation_driver"] == "through_conductivity_only"
    assert summary["primary_fraction_of_all_property_attenuation"] > 1.0
    assert summary["primary_driver_resolved_by_frozen_criterion"] is True
    assert summary["two_factor_interaction_panel_authorized"] is False
    assert summary["noise_confirmation_authorized"] is False
    factors = payload["one_factor_results"]
    assert factors["density_only"]["meets_observation_threshold"] is True
    assert factors["axial_conductivity_only"]["meets_observation_threshold"] is True
    assert factors["through_conductivity_only"][
        "meets_observation_threshold"
    ] is False
    assert 0.84 < factors["through_conductivity_only"][
        "rms_signal_to_declared_noise"
    ] < 0.86


def test_r128_profiles_the_through_conductivity_observation_boundary():
    payload = _load(
        "artifacts/paper_r128_through_conductivity_profile/development.json"
    )
    assert payload["observation_and_mission_frozen_from"] == "R124"
    assert payload["nuisance_axis_frozen_from"] == "R127"
    assert payload["endpoint_interval_frozen_from"] == "R100"
    assert payload["active_interventions_used"] is False
    assert payload["expected_new_interior_runs"] == 6
    assert payload["complete_cached_or_generated_interior_runs"] == 6
    assert payload["endpoint_runs_reused"] == 4
    assert payload["all_runs_finite"] is True
    profile = payload["profile"]
    conductivity = [row["conductivity_through_w_per_m_k"] for row in profile]
    assert conductivity == [9.0, 9.925, 10.85, 11.774999999999999, 12.7]
    assert profile[0]["meets_observation_threshold"] is True
    assert all(
        row["meets_observation_threshold"] is False for row in profile[1:]
    )
    summary = payload["summary"]
    assert summary["rms_strictly_decreases_over_profile"] is True
    assert 9.7 < summary[
        "linear_interpolated_conductivity_at_rms_one_w_per_m_k"
    ] < 9.8
    assert summary["profile_brackets_observation_threshold"] is True
    assert summary["material_nuisance_profile_ready_for_candidate_scoring"] is True
    assert summary["noise_confirmation_authorized"] is False


def test_r129_material_profiled_candidate_forward_preserves_truth_but_is_broad():
    payload = _load(
        "artifacts/paper_r129_material_profiled_candidate_forward/development.json"
    )
    assert payload["observation_and_mission_frozen_from"] == "R124"
    assert payload["material_nodes_frozen_from"] == "R128"
    assert payload["active_interventions_used"] is False
    assert payload["forchheimer_trajectory_runs_reused"] == 10
    assert payload["expected_new_darcy_runs"] == 10
    assert payload["complete_cached_or_generated_darcy_runs"] == 10
    assert payload["all_runs_finite"] is True
    summary = payload["summary"]
    assert summary["truth_panel_count"] == 20
    assert summary["truth_structure_coverage_count"] == 20
    assert summary["unique_truth_structure_count"] == 1
    assert summary["all_truth_structures_covered"] is True
    assert summary["all_truth_structures_unique"] is False
    assert summary["noise_confirmation_authorized"] is False


def test_r130_passive_multichannel_tiers_reduce_but_do_not_remove_ambiguity():
    payload = _load(
        "artifacts/paper_r130_multichannel_material_profiled_adequacy/development.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["active_interventions_used"] is False
    assert payload["material_node_must_be_common_across_blocks"] is True
    tiers = payload["tier_summary"]
    assert tiers["focused_product_only"]["unique_truth_structure_count"] == 1
    assert tiers["focused_three_species"]["unique_truth_structure_count"] == 11
    assert tiers["focused_three_species_plus_standard_ground"][
        "unique_truth_structure_count"
    ] == 11
    assert tiers["focused_research_multiphysics"][
        "unique_truth_structure_count"
    ] == 12
    assert all(row["truth_structure_coverage_count"] == 20 for row in tiers.values())
    summary = payload["summary"]
    assert summary["best_tier"] == "focused_research_multiphysics"
    assert summary["best_tier_mean_structure_set_width"] == 1.4
    assert summary["all_truth_structures_unique_in_best_tier"] is False
    assert summary["noise_confirmation_authorized"] is False


def test_r131_freezes_a_twenty_percent_product_precision_requirement():
    payload = _load(
        "artifacts/paper_r131_product_precision_contract/development.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["noise_seeds_used"] is False
    assert payload["active_interventions_used"] is False
    assert payload["precision_changed_only_for_block"] == (
        "inner_surface_microports_pProducts"
    )
    ladder = payload["ladder_summary"]
    assert [ladder[key]["unique_truth_structure_count"] for key in (
        "1.00", "1.10", "1.20", "1.25"
    )] == [12, 16, 20, 20]
    summary = payload["summary"]
    assert summary["selected_minimum_ladder_precision_factor"] == 1.2
    assert summary["selected_equivalent_product_noise_multiplier"] == (
        0.8333333333333334
    )
    assert summary["clean_confirmation_layout_authorized"] is True
    assert summary["noisy_confirmation_run_in_this_experiment"] is False


def test_r132_correlated_noise_preserves_coverage_but_not_unique_structure():
    payload = _load(
        "artifacts/paper_r132_correlated_noise_confirmation/confirmation.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["active_interventions_used"] is False
    assert payload["product_precision_factor"] == 1.2
    assert payload["confirmation_cases_per_noise_model"] == 400
    calibration = payload["calibration"]
    assert 1.8 < calibration["iid_gaussian"]["threshold"] < 2.0
    assert 2.2 < calibration["ar1_rho_0p8"]["threshold"] < 2.4
    assert 2.2 < calibration[
        "mixed_block_bias_fraction_0p5"
    ]["threshold"] < 2.4
    aggregate = payload["aggregate_by_noise_model"]
    assert 0.93 < aggregate["iid_gaussian"]["truth_coverage_rate"] < 0.96
    assert 0.93 < aggregate["ar1_rho_0p8"]["truth_coverage_rate"] < 0.96
    assert 0.94 < aggregate[
        "mixed_block_bias_fraction_0p5"
    ]["truth_coverage_rate"] < 0.97
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.47
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.26
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.245
    assert aggregate["iid_gaussian"]["wrong_only_count"] == 0
    assert aggregate["ar1_rho_0p8"]["wrong_only_count"] == 0
    assert aggregate["mixed_block_bias_fraction_0p5"]["wrong_only_count"] == 1
    summary = payload["summary"]
    assert all(passed is False for passed in summary["noise_model_pass"].values())
    assert summary["all_noise_models_pass"] is False
    assert summary[
        "material_profiled_noisy_structure_confirmation_supported"
    ] is False


def test_r133_noise_aware_tier_development_selects_three_species_but_withholds_confirmation():
    payload = _load(
        "artifacts/paper_r133_noise_aware_tier_development/development.json"
    )
    assert payload["role"] == "disjoint_seed_tier_development_not_confirmation"
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["active_interventions_used"] is False
    assert payload["r132_confirmation_seeds_reused"] is False
    aggregates = payload["aggregate_by_tier_and_noise"]
    focused = aggregates["focused_three_species"]
    assert focused["iid_gaussian"]["truth_coverage_rate"] == 0.98
    assert focused["ar1_rho_0p8"]["truth_coverage_rate"] == 0.96
    assert focused["mixed_block_bias_fraction_0p5"][
        "truth_coverage_rate"
    ] == 0.9575
    assert focused["iid_gaussian"]["unique_truth_rate"] == 0.55
    assert focused["ar1_rho_0p8"]["unique_truth_rate"] == 0.335
    assert focused["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.3575
    summary = payload["summary"]
    assert summary["selected_tier"] == "focused_three_species"
    assert summary["selected_tier_worst_noise_unique_truth_rate"] == 0.335
    assert summary["selected_tier_meets_unique_target"] is False
    assert summary["fresh_confirmation_authorized"] is False


def test_r134_covariance_joint_scoring_improves_width_but_remains_development_only():
    payload = _load(
        "artifacts/paper_r134_covariance_aware_joint_scoring/development.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["tier"] == "focused_three_species"
    assert payload["r132_confirmation_seeds_reused"] is False
    aggregate = payload["aggregate_by_method_and_noise"]["covariance_aware_gls"]
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.635
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.6925
    assert aggregate["ar1_rho_0p8"]["mean_structure_set_width"] == 1.325
    assert payload["summary"]["fresh_confirmation_authorized"] is False


def test_r135_four_passive_repeats_are_insufficient():
    payload = _load("artifacts/paper_r135_passive_replication_budget/development.json")
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["active_interventions_used"] is False
    assert payload["calibration_quantile"] == 0.98
    assert payload["calibration_quantile_derived_from_wrong_only_ceiling"] is True
    aggregate = payload["aggregate_by_repeat_count_and_noise"]["4"]
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.805
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.71
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.705
    assert payload["summary"]["selected_minimum_repeat_count"] is None
    assert payload["summary"]["fresh_confirmation_authorized"] is False


def test_r136_fresh_nested_development_selects_sixteen_repeats():
    payload = _load(
        "artifacts/paper_r136_conservative_replication_development/development.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["nested_repeat_ladder"] is True
    assert payload["repeat_counts"] == [4, 8, 12, 16]
    assert payload["calibration_quantile"] == 0.99
    assert payload["r133_r135_development_seeds_reused"] is False
    assert payload["repeat_contract"]["12"]["all_noise_models_pass"] is False
    assert payload["repeat_contract"]["16"]["all_noise_models_pass"] is True
    aggregate = payload["aggregate_by_repeat_count_and_noise"]["16"]
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.965
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.86
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.89
    assert payload["summary"]["selected_minimum_repeat_count"] == 16
    assert payload["summary"]["fresh_confirmation_authorized"] is True


def test_r137_frozen_replicated_joint_scoring_confirms_all_noise_models():
    payload = _load(
        "artifacts/paper_r137_replicated_joint_scoring_confirmation/confirmation.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["calibration_rerun_or_modified"] is False
    assert payload["development_seeds_reused"] is False
    assert payload["repeat_count"] == 16
    assert payload["confirmation_cases_per_noise_model"] == 400
    aggregate = payload["aggregate_by_noise_model"]
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.9575
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.84
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.9025
    assert all(row["truth_coverage_rate"] >= 0.99 for row in aggregate.values())
    assert all(row["wrong_only_rate"] <= 0.01 for row in aggregate.values())
    summary = payload["summary"]
    assert all(summary["noise_model_pass"].values())
    assert summary["all_noise_models_pass"] is True
    assert summary["replicated_noisy_structure_identification_supported"] is True


def test_r138_offgrid_material_truth_exposes_discrete_profile_bias():
    payload = _load(
        "artifacts/paper_r138_offgrid_material_confirmation/confirmation.json"
    )
    assert payload["truth_conductivities_are_candidate_nodes"] is False
    assert payload["full_coupled_truth_run_count"] == 16
    assert payload["truth_trajectories_generated_by_r138"] is True
    assert payload["all_runs_finite"] is True
    aggregate = payload["aggregate_by_noise_model"]
    assert aggregate["iid_gaussian"]["truth_coverage_rate"] == 0.7125
    assert aggregate["ar1_rho_0p8"]["truth_coverage_rate"] == 0.928125
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "truth_coverage_rate"
    ] == 0.846875
    summary = payload["summary"]
    assert all(passed is False for passed in summary["noise_model_pass"].values())
    assert summary["offgrid_material_structure_identification_supported"] is False


def test_r139_continuous_profile_is_accurate_but_sixteen_repeats_are_insufficient():
    payload = _load(
        "artifacts/paper_r139_continuous_material_profile_development/development.json"
    )
    assert payload["r138_offgrid_truths_loaded"] is False
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["maximum_leave_one_node_rms_error_to_truth_noise"] < 0.09
    aggregate = payload["aggregate_by_noise_model"]
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.86
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.7525
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.8
    assert payload["summary"]["fresh_offgrid_confirmation_authorized"] is False


def test_r140_continuous_profile_selects_twenty_four_repeats_and_stops():
    payload = _load(
        "artifacts/paper_r140_continuous_profile_replication_budget/development.json"
    )
    assert payload["r138_offgrid_truths_loaded"] is False
    assert payload["repeat_counts_predeclared"] == [24, 32]
    assert payload["stop_at_first_passing_count"] is True
    assert set(payload["aggregate_by_repeat_count_and_noise"]) == {"24"}
    aggregate = payload["aggregate_by_repeat_count_and_noise"]["24"]
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.905
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.8225
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.8525
    assert payload["summary"]["selected_minimum_repeat_count"] == 24
    assert payload["summary"]["fresh_offgrid_confirmation_authorized"] is True
    assert payload["summary"]["replication_search_terminated"] is True


def test_r141_frozen_continuous_profile_confirms_offgrid_material_truths():
    payload = _load(
        "artifacts/paper_r141_offgrid_continuous_profile_confirmation/confirmation.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["r138_truth_trajectories_reused"] == 16
    assert payload["calibration_rerun_or_modified"] is False
    assert payload["development_seeds_reused"] is False
    assert payload["repeat_count"] == 24
    assert payload["confirmation_cases_per_noise_model"] == 320
    aggregate = payload["aggregate_by_noise_model"]
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.91875
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.81875
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.859375
    assert all(row["truth_coverage_rate"] >= 0.99 for row in aggregate.values())
    assert all(row["wrong_only_rate"] <= 0.01 for row in aggregate.values())
    summary = payload["summary"]
    assert all(summary["noise_model_pass"].values())
    assert summary["all_noise_models_pass"] is True
    assert summary["offgrid_continuous_material_identification_supported"] is True


def test_r142_independent_gas_preflight_rejects_single_solver_forchheimer_claim():
    payload = _load(
        "artifacts/paper_r142_independent_gas_operator_preflight/preflight.json"
    )
    assert payload["full_seven_field_cross_solver"] is False
    assert payload["all_runs_finite"] is True
    discrepancy = payload["cross_solver_block_rms_discrepancy_to_declared_noise"]
    assert discrepancy["Darcy__isotropic"][
        "inner_surface_microports_pH2O"
    ] < 0.25
    assert discrepancy["Forchheimer__directional"][
        "inner_surface_microports_pH2O"
    ] > 12.0
    summary = payload["summary"]
    assert summary["clean_case_count"] == 12
    assert summary["clean_truth_coverage_count"] == 6
    assert summary["clean_unique_truth_count"] == 6
    assert summary["all_clean_truths_covered"] is False
    assert summary["full_noisy_cross_solver_panel_authorized"] is False


def test_r143_bracketing_solver_ensemble_restores_clean_adequacy():
    payload = _load(
        "artifacts/paper_r143_solver_ensemble_clean_adequacy/development.json"
    )
    assert payload["full_seven_field_cross_solver"] is False
    assert payload["independent_candidate_bracketing_nodes_w_per_m_k"] == [
        9.925,
        10.85,
    ]
    assert payload["all_runs_finite"] is True
    summary = payload["summary"]
    assert summary["clean_case_count"] == 12
    assert summary["clean_truth_coverage_count"] == 12
    assert summary["clean_unique_truth_count"] == 12
    assert summary["full_solver_ensemble_development_authorized"] is True


def test_r144_full_solver_ensemble_passes_every_truth_source_and_noise_group():
    payload = _load(
        "artifacts/paper_r144_full_solver_ensemble_development/development.json"
    )
    assert payload["r142_offgrid_truths_loaded"] is False
    assert payload["candidate_solver_forms"] == ["primary", "independent_gas"]
    assert payload["truth_solver_forms"] == ["primary", "independent_gas"]
    assert payload["full_seven_field_cross_solver"] is False
    assert payload["r143_candidate_runs_reused"] == 8
    assert payload["all_runs_finite"] is True
    aggregate = payload["aggregate_by_noise_model_and_truth_solver"]
    assert aggregate["iid_gaussian"]["primary"]["unique_truth_rate"] == 0.885
    assert aggregate["iid_gaussian"]["independent_gas"][
        "unique_truth_rate"
    ] == 0.8325
    assert aggregate["ar1_rho_0p8"]["primary"]["unique_truth_rate"] == 0.845
    assert aggregate["ar1_rho_0p8"]["independent_gas"][
        "unique_truth_rate"
    ] == 0.9175
    assert all(
        passed
        for by_solver in payload["group_pass"].values()
        for passed in by_solver.values()
    )
    assert payload["summary"][
        "fresh_independent_offgrid_confirmation_authorized"
    ] is True


def test_r145_frozen_solver_ensemble_confirms_independent_offgrid_truths():
    payload = _load(
        "artifacts/paper_r145_independent_offgrid_solver_ensemble_confirmation/confirmation.json"
    )
    assert payload["full_seven_field_cross_solver"] is False
    assert payload["candidate_solver_forms"] == ["primary", "independent_gas"]
    assert payload["r142_truth_runs_reused"] == 4
    assert payload["all_runs_finite"] is True
    assert payload["calibration_rerun_or_modified"] is False
    assert payload["development_seeds_reused"] is False
    assert payload["confirmation_cases_per_noise_model"] == 320
    aggregate = payload["aggregate_by_noise_model"]
    assert aggregate["iid_gaussian"]["unique_truth_rate"] == 0.846875
    assert aggregate["ar1_rho_0p8"]["unique_truth_rate"] == 0.93125
    assert aggregate["mixed_block_bias_fraction_0p5"][
        "unique_truth_rate"
    ] == 0.921875
    assert all(row["truth_coverage_rate"] >= 0.99 for row in aggregate.values())
    assert all(row["wrong_only_rate"] < 0.01 for row in aggregate.values())
    summary = payload["summary"]
    assert all(summary["noise_model_pass"].values())
    assert summary["all_noise_models_pass"] is True
    assert summary["independent_gas_offgrid_identification_supported"] is True


def test_r146_clean_transfer_isolates_geometry_conditioning_failure():
    payload = _load(
        "artifacts/paper_r146_process_variation_clean_transfer/development.json"
    )
    assert payload["new_forward_runs"] == 0
    aggregate = payload["aggregate_by_physical_case"]
    assert aggregate["reference_bartz"]["truth_coverage_count"] == 6
    assert aggregate["material_batch_shift"]["truth_coverage_count"] == 6
    assert aggregate["initial_defect_shift"]["truth_coverage_count"] == 6
    assert aggregate["geometry_envelope_shift"]["truth_coverage_count"] == 3
    assert aggregate["combined_shift"]["truth_coverage_count"] == 3
    assert payload["summary"]["clean_wrong_only_count"] == 6
    assert payload["summary"]["noisy_process_variation_panel_authorized"] is False


def test_r147_known_geometry_restores_every_clean_transfer_case():
    payload = _load(
        "artifacts/paper_r147_geometry_conditioned_candidate_forward/development.json"
    )
    assert payload[
        "geometry_is_observed_condition_not_profiled_hidden_mechanism"
    ] is True
    assert len(payload["primary_generation"]) == 20
    assert len(payload["independent_generation"]) == 20
    assert payload["summary"]["clean_truth_coverage_count"] == 30
    assert payload["summary"]["clean_unique_truth_count"] == 30
    assert payload["summary"]["clean_wrong_only_count"] == 0
    assert payload["summary"]["noisy_geometry_conditioned_panel_authorized"] is True


def test_r148_twenty_four_repeat_process_transfer_is_not_uniformly_robust():
    payload = _load(
        "artifacts/paper_r148_noisy_process_variation_confirmation/confirmation.json"
    )
    assert payload["repeat_count_per_identification"] == 24
    assert payload["summary"]["confirmation_case_count"] == 6000
    assert payload["summary"]["all_physical_case_and_noise_groups_pass"] is False
    geometry_ar1 = payload["aggregate_by_case_and_noise"][
        "geometry_envelope_shift"
    ]["ar1_rho_0p8"]
    assert geometry_ar1["wrong_only_rate"] == 0.045


def test_r149_conditional_thresholds_protect_coverage_but_leave_broad_sets():
    payload = _load(
        "artifacts/paper_r149_structure_conditional_threshold_development/development.json"
    )
    assert payload["truth_label_required_at_inference"] is False
    assert payload["summary"]["fresh_confirmation_authorized"] is False
    reference_ar1 = payload["aggregate_by_case_and_noise"]["reference_bartz"][
        "ar1_rho_0p8"
    ]
    assert reference_ar1["truth_coverage_rate"] == 0.99
    assert reference_ar1["unique_truth_rate"] == 0.77


def test_r150_three_species_joint_is_best_topology_channel_tier():
    payload = _load(
        "artifacts/paper_r150_topology_channel_attribution/development.json"
    )
    assert payload["new_forward_runs"] == 0
    summary = payload["summary"]
    assert summary["selected_topology_tier"] == "three_species_joint"
    assert summary["worst_case_noise_topology_accuracy"] == 0.91
    assert summary["two_stage_method_development_authorized"] is True
    metrics = payload["selection_metrics"]
    assert metrics["three_species_joint"][
        "worst_case_noise_topology_accuracy"
    ] > metrics["pProducts_only"]["worst_case_noise_topology_accuracy"]


def test_r151_global_32_repeat_rule_remains_geometry_limited():
    payload = _load(
        "artifacts/paper_r151_process_robust_replication_development/development.json"
    )
    assert payload["repeat_count"] == 32
    assert payload["development_replicates_per_case_topology_noise"] == 200
    assert payload["summary"]["fresh_large_sample_confirmation_authorized"] is False
    geometry_ar1 = payload["aggregate_by_case_and_noise"][
        "geometry_envelope_shift"
    ]["ar1_rho_0p8"]
    assert geometry_ar1["wrong_only_rate"] == 0.0325


def test_r152_conditional_32_repeat_rule_still_has_reference_ambiguity():
    payload = _load(
        "artifacts/paper_r152_conditional_32repeat_development/development.json"
    )
    assert payload["repeat_count"] == 32
    assert payload["truth_label_required_at_inference"] is False
    assert payload["summary"]["fresh_large_sample_confirmation_authorized"] is False
    reference_ar1 = payload["aggregate_by_case_and_noise"]["reference_bartz"][
        "ar1_rho_0p8"
    ]
    assert reference_ar1["unique_truth_rate"] == 0.7575


def test_r153_standard_48_repeat_rule_misses_one_group_by_one_case():
    payload = _load(
        "artifacts/paper_r153_48repeat_process_robust_development/development.json"
    )
    assert payload["repeat_count"] == 48
    assert payload["summary"]["fresh_large_sample_confirmation_authorized"] is False
    initial_ar1 = payload["aggregate_by_case_and_noise"][
        "initial_defect_shift"
    ]["ar1_rho_0p8"]
    assert initial_ar1["wrong_only_count"] == 9
    assert initial_ar1["wrong_only_rate"] == 0.0225


def test_r154_conservative_48_repeat_rule_passes_development():
    payload = _load(
        "artifacts/paper_r154_conservative_48repeat_development/development.json"
    )
    assert payload["repeat_count"] == 48
    assert all(
        calibration["quantile"] == 0.995
        for by_noise in payload["calibration"].values()
        for calibration in by_noise.values()
    )
    assert all(
        passed
        for by_noise in payload["group_pass"].values()
        for passed in by_noise.values()
    )
    assert payload["summary"]["fresh_large_sample_confirmation_authorized"] is True


def test_r155_frozen_confirmation_records_single_near_boundary_failure():
    payload = _load(
        "artifacts/paper_r155_process_robust_confirmation/confirmation.json"
    )
    assert payload["method_frozen_from"] == "R154"
    assert payload["repeat_count"] == 48
    assert payload["calibration_draws_per_geometry_noise"] == 5000
    assert payload["calibration_quantile"] == 0.995
    assert payload["summary"]["confirmation_case_count"] == 6000
    failed = [
        (case_id, model)
        for case_id, by_noise in payload["group_pass"].items()
        for model, passed in by_noise.items()
        if not passed
    ]
    assert failed == [("geometry_envelope_shift", "ar1_rho_0p8")]
    geometry_ar1 = payload["aggregate_by_case_and_noise"][
        "geometry_envelope_shift"
    ]["ar1_rho_0p8"]
    assert geometry_ar1["truth_coverage_rate"] == 0.9775
    assert geometry_ar1["unique_truth_rate"] == 0.975
    assert geometry_ar1["wrong_only_count"] == 9
    assert geometry_ar1["wrong_only_rate"] == 0.0225
    assert payload["summary"]["all_physical_case_and_noise_groups_pass"] is False


def test_r156_full_conditional_rule_fixes_geometry_but_broadens_reference_sets():
    payload = _load(
        "artifacts/paper_r156_conditional_48repeat_development/development.json"
    )
    assert payload["repeat_count"] == 48
    assert payload["r155_confirmation_rows_used_for_tuning"] is False
    assert payload["truth_label_required_at_inference"] is False
    geometry_ar1 = payload["aggregate_by_case_and_noise"][
        "geometry_envelope_shift"
    ]["ar1_rho_0p8"]
    reference_ar1 = payload["aggregate_by_case_and_noise"]["reference_bartz"][
        "ar1_rho_0p8"
    ]
    initial_ar1 = payload["aggregate_by_case_and_noise"][
        "initial_defect_shift"
    ]["ar1_rho_0p8"]
    assert geometry_ar1["wrong_only_rate"] == 0.0025
    assert reference_ar1["unique_truth_rate"] == 0.7
    assert initial_ar1["unique_truth_rate"] == 0.7275
    assert payload["summary"]["fresh_confirmation_authorized"] is False


def test_r157_directional_guard_restores_width_but_not_geometry_ar1_error():
    payload = _load(
        "artifacts/paper_r157_directional_guard_development/development.json"
    )
    assert payload["repeat_count"] == 48
    assert payload["r155_confirmation_rows_used_for_tuning"] is False
    assert payload["guarded_candidate_structure"] == "Forchheimer__directional"
    reference_ar1 = payload["aggregate_by_case_and_noise"]["reference_bartz"][
        "ar1_rho_0p8"
    ]
    geometry_ar1 = payload["aggregate_by_case_and_noise"][
        "geometry_envelope_shift"
    ]["ar1_rho_0p8"]
    assert reference_ar1["unique_truth_rate"] == 0.8475
    assert geometry_ar1["wrong_only_count"] == 9
    assert geometry_ar1["wrong_only_rate"] == 0.0225
    assert payload["summary"]["fresh_confirmation_authorized"] is False


def test_r158_candidate_physics_weighting_is_rejected_against_paired_equal_gls():
    payload = _load(
        "artifacts/paper_r158_physics_weighted_channel_development/development.json"
    )
    assert payload["new_forward_runs"] == 0
    assert payload[
        "r155_confirmation_rows_used_for_weight_or_threshold_selection"
    ] is False
    assert payload["paired_observations_across_methods"] is True
    summary = payload["summary"]
    assert summary["development_identification_count"] == 12000
    assert summary["method_all_groups_pass"]["equal_gls"] is True
    assert summary["method_all_groups_pass"][
        "candidate_physics_weighted_gls"
    ] is False
    assert summary["weighted_method_fresh_confirmation_authorized"] is False
    equal_geometry_ar1 = payload["aggregate_by_method_case_noise"]["equal_gls"][
        "geometry_envelope_shift"
    ]["ar1_rho_0p8"]
    weighted_geometry_ar1 = payload["aggregate_by_method_case_noise"][
        "candidate_physics_weighted_gls"
    ]["geometry_envelope_shift"]["ar1_rho_0p8"]
    assert equal_geometry_ar1["wrong_only_rate"] == 0.0175
    assert weighted_geometry_ar1["wrong_only_rate"] == 0.0275
    weights = payload["physics_weight_records"]["shifted_geometry"][
        "ar1_rho_0p8"
    ]["weights"]
    values = list(weights.values())
    effective_channels = sum(values) ** 2 / sum(value**2 for value in values)
    assert effective_channels < 1.20


def test_r159_completes_darcy_truths_and_passes_clean_gate():
    payload = _load(
        "artifacts/paper_r159_darcy_process_variation_clean_transfer/development.json"
    )
    assert payload["truth_law"] == "Darcy"
    assert payload["new_full_coupled_runs"] == 6
    assert payload["reused_full_coupled_runs"] == 4
    assert all(row["all_fields_finite"] for row in payload["generation"])
    summary = payload["summary"]
    assert summary["clean_case_count"] == 30
    assert summary["clean_truth_coverage_count"] == 30
    assert summary["clean_unique_truth_count"] == 30
    assert summary["clean_unique_best_count"] == 30
    assert summary["clean_wrong_only_count"] == 0
    assert summary["noisy_darcy_variation_development_authorized"] is True


def test_r160_noisy_darcy_development_authorizes_one_confirmation():
    payload = _load(
        "artifacts/paper_r160_noisy_darcy_process_variation_development/development.json"
    )
    assert payload["truth_law"] == "Darcy"
    assert payload["repeat_count"] == 48
    assert payload["new_forward_runs"] == 0
    assert payload["summary"]["development_case_count"] == 6000
    assert all(
        passed
        for by_noise in payload["group_pass"].values()
        for passed in by_noise.values()
    )
    assert payload["summary"]["fresh_darcy_confirmation_authorized"] is True


def test_r161_frozen_darcy_confirmation_is_perfect_across_declared_shifts():
    payload = _load(
        "artifacts/paper_r161_darcy_process_variation_confirmation/confirmation.json"
    )
    assert payload["method_frozen_from"] == "R160 using R155 calibration"
    assert payload["calibration_rerun_or_modified"] is False
    assert payload["development_seeds_reused"] is False
    assert payload["summary"]["confirmation_case_count"] == 6000
    assert payload["summary"]["darcy_physical_variation_confirmed"] is True
    aggregates = [
        record
        for by_noise in payload["aggregate_by_case_and_noise"].values()
        for record in by_noise.values()
    ]
    assert all(record["truth_coverage_rate"] == 1.0 for record in aggregates)
    assert all(record["unique_truth_rate"] == 1.0 for record in aggregates)
    assert all(record["wrong_only_rate"] == 0.0 for record in aggregates)


def test_r162_clean_readout_state_drift_preserves_unique_structure():
    payload = _load(
        "artifacts/paper_r162_readout_state_drift_clean_gate/development.json"
    )
    assert payload["active_interventions_used"] is False
    assert payload["repeat_count"] == 48
    assert payload["new_forward_runs"] == 0
    assert payload["maximum_drift_fractions"] == [0.0, 0.1, 0.25, 0.5, 1.0]
    summary = payload["summary"]
    assert summary["clean_case_count"] == 180
    assert summary["largest_all_unique_maximum_drift_fraction"] == 1.0
    assert summary["all_fractions_unique"] is True
    assert summary["noisy_state_drift_development_authorized"] is True


def test_r163_noisy_state_drift_preserves_near_boundary_development_result():
    payload = _load(
        "artifacts/paper_r163_noisy_readout_state_drift_development/development.json"
    )
    assert payload["active_interventions_used"] is False
    assert payload["new_forward_runs"] == 0
    assert payload["repeat_count"] == 48
    assert payload["drift_fractions"] == [0.0, 0.1, 0.25]
    assert payload["summary"]["development_case_count"] == 21600
    assert payload["fraction_pass"] == {
        "0.0": False,
        "0.1": True,
        "0.25": False,
    }
    assert payload["summary"]["primary_10percent_all_groups_pass"] is True
    assert payload["summary"]["stress_25percent_all_groups_pass"] is False
    assert payload["summary"]["fresh_10percent_drift_confirmation_authorized"] is True

    aggregates = payload["aggregate_by_fraction_axis_family_noise"]
    assert (
        aggregates["0.0"]["material_defect_at_shifted_geometry"]
        ["Forchheimer"]["ar1_rho_0p8"]["wrong_only_rate"]
        == 0.0325
    )
    assert (
        aggregates["0.25"]["material_defect_at_shifted_geometry"]
        ["Forchheimer"]["ar1_rho_0p8"]["wrong_only_rate"]
        == 0.0225
    )


def test_r164_frozen_ten_percent_state_drift_confirmation_passes_all_groups():
    payload = _load(
        "artifacts/paper_r164_readout_state_drift_confirmation/confirmation.json"
    )
    assert payload["method_frozen_from"] == "R163 primary 10% drift development"
    assert payload["development_seeds_reused"] is False
    assert payload["calibration_rerun_or_modified"] is False
    assert payload["active_interventions_used"] is False
    assert payload["new_forward_runs"] == 0
    assert payload["maximum_drift_fraction"] == 0.1
    summary = payload["summary"]
    assert summary["confirmation_case_count"] == 7200
    assert summary["passing_group_count"] == 18
    assert summary["group_count"] == 18
    assert summary["ten_percent_readout_state_drift_confirmed"] is True

    records = [
        record
        for by_family in payload["aggregate_by_axis_family_noise"].values()
        for by_noise in by_family.values()
        for record in by_noise.values()
    ]
    assert min(record["truth_coverage_rate"] for record in records) == 0.9825
    assert min(record["unique_truth_rate"] for record in records) == 0.8875
    assert max(record["wrong_only_rate"] for record in records) == 0.0175


def test_r165_geometry_metadata_tangent_preflight_requires_exact_candidates():
    payload = _load(
        "artifacts/paper_r165_geometry_metadata_uncertainty_preflight/preflight.json"
    )
    assert payload["active_interventions_used"] is False
    assert payload["new_forward_runs"] == 0
    assert payload["repeat_count"] == 48
    assert payload["geometry_bias_fractions_of_full_engineering_shift"] == [
        0.0,
        0.01,
        0.05,
        0.1,
    ]
    assert payload["all_unique_by_bias_fraction"] == {
        "0.0": True,
        "0.01": True,
        "0.05": False,
        "0.1": False,
    }
    summary = payload["summary"]
    assert summary["clean_case_count"] == 240
    assert summary["zero_bias_control_all_unique"] is True
    assert summary["largest_all_unique_bias_fraction"] == 0.01
    assert summary["exact_perturbed_geometry_candidate_runs_required"] is True

    five_percent = payload["aggregate_by_bias_fraction_and_truth_case"]["0.05"]
    assert five_percent["geometry_envelope_shift"] == {
        "case_count": 12,
        "truth_coverage_count": 9,
        "unique_truth_count": 9,
        "wrong_only_count": 3,
    }


def test_r166_exact_geometry_attribution_isolates_expansion_ratio():
    payload = _load(
        "artifacts/paper_r166_exact_single_parameter_geometry_attribution/attribution.json"
    )
    assert payload["active_interventions_used"] is False
    assert payload["full_coupled_primary_candidate_trajectory_count"] == 40
    assert payload["independent_gas_candidate_trajectory_count"] == 40
    assert payload["generated_primary_runs_this_invocation"] == 0
    assert payload["generated_independent_runs_this_invocation"] == 0
    assert payload["summary"] == {
        "clean_attribution_case_count": 30,
        "baseline_exact_geometry_all_unique": True,
        "single_parameter_failure_modes": ["expansion_ratio"],
        "coupled_geometry_interaction_suspected": False,
        "full_family_noisy_extension_authorized_modes": [],
        "uncertainty_aware_geometry_profiling_required_modes": [
            "expansion_ratio"
        ],
    }
    aggregate = payload["aggregate_by_geometry_mode"]
    for mode in (
        "baseline_exact_geometry",
        "length",
        "wall_thickness",
        "exit_angle",
    ):
        assert aggregate[mode]["unique_truth_count"] == 6
        assert aggregate[mode]["wrong_only_count"] == 0
    assert aggregate["expansion_ratio"]["truth_coverage_count"] == 5
    assert aggregate["expansion_ratio"]["unique_truth_count"] == 5
    assert aggregate["expansion_ratio"]["wrong_only_count"] == 1


def test_r167_expansion_ratio_profile_repairs_clean_topology_alias():
    payload = _load(
        "artifacts/paper_r167_expansion_ratio_profile_clean_gate/development.json"
    )
    assert payload["active_interventions_used"] is False
    assert payload["truth_full_coupled_trajectory_count"] == 6
    assert payload["generated_truth_runs_this_invocation"] == 0
    assert payload["truth_fractions"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert payload["summary"] == {
        "clean_case_count_per_method": 30,
        "fixed_reported_geometry_all_unique": False,
        "profiled_uncertainty_bracket_all_unique": True,
        "maximum_truth_geometry_fraction_error": 0.0,
        "noisy_development_authorized": True,
    }
    assert payload["aggregate_by_method"]["fixed_reported_upper_endpoint"] == {
        "case_count": 30,
        "truth_coverage_count": 29,
        "unique_truth_count": 29,
        "wrong_only_count": 1,
    }
    assert payload["aggregate_by_method"]["profiled_uncertainty_bracket"] == {
        "case_count": 30,
        "truth_coverage_count": 30,
        "unique_truth_count": 30,
        "wrong_only_count": 0,
    }


def test_r168_24repeat_diagnostic_improves_coverage_but_not_uniqueness():
    payload = _load(
        "artifacts/paper_r168_noisy_expansion_ratio_profile_development/development.json"
    )
    assert payload["protocol"] == (
        "r168-v2-995-recalibrated-noisy-expansion-ratio-profile"
    )
    assert payload["active_interventions_used"] is False
    assert payload["repeat_count"] == 24
    assert payload["calibration_quantile"] == 0.995
    assert payload["summary"] == {
        "development_case_count": 12000,
        "profiled_method_all_groups_pass": False,
        "frozen_confirmation_authorized": False,
    }
    fixed = payload["method_summary"]["fixed_reported_upper_endpoint"]
    profiled = payload["method_summary"]["profiled_uncertainty_bracket"]
    assert fixed["passing_group_count"] == 16
    assert fixed["minimum_truth_coverage_rate"] == 0.545
    assert fixed["minimum_unique_truth_rate"] == 0.165
    assert fixed["maximum_wrong_only_rate"] == 0.455
    assert profiled["passing_group_count"] == 18
    assert profiled["minimum_truth_coverage_rate"] == 0.97
    assert profiled["minimum_unique_truth_rate"] == 0.495
    assert profiled["maximum_wrong_only_rate"] == 0.03
    failed = []
    for model, by_fraction in payload["group_pass"][
        "profiled_uncertainty_bracket"
    ].items():
        for fraction, by_structure in by_fraction.items():
            for structure, passed in by_structure.items():
                if not passed:
                    failed.append((model, fraction, structure))
    assert len(failed) == 12
    assert all(
        structure == "Forchheimer__directional"
        for _, _, structure in failed
    )


def test_r168_corrected_48repeat_result_keeps_the_failed_group_visible():
    payload = _load(
        "artifacts/paper_r168_noisy_expansion_ratio_profile_development/development_48repeat.json"
    )
    assert payload["protocol"] == (
        "r168-v3-48repeat-995-recalibrated-noisy-expansion-ratio-profile"
    )
    assert payload["repeat_count"] == 48
    assert payload["summary"]["frozen_confirmation_authorized"] is False
    assert len(payload["rows"]) == 12000
    failed = [
        (model, fraction, structure)
        for model, by_fraction in payload["group_pass"][
            "profiled_uncertainty_bracket"
        ].items()
        for fraction, by_structure in by_fraction.items()
        for structure, passed in by_structure.items()
        if not passed
    ]
    assert failed == [
        ("mixed_block_bias_fraction_0p5", "0.25", "Forchheimer__directional")
    ]
    rows = [
        row for row in payload["rows"]
        if row["method"] == "profiled_uncertainty_bracket"
        and row["noise_model"] == failed[0][0]
        and row["truth_fraction"] == 0.25
        and row["truth_structure"] == failed[0][2]
    ]
    assert len(rows) == 200
    assert sum(row["wrong_only"] for row in rows) == 5
