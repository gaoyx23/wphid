"""Build the paper-facing WPHID artifact and environment manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "nozzle_truth_qualification": "artifacts/paper_r2_geometry_resolved_nozzle_qualification/metrics.json",
    "passive_nozzle_dataset": "artifacts/paper_r3_passive_nozzle_dataset/manifest.json",
    "reality_constrained_conditions": "artifacts/paper_r5_reality_constrained_natural_conditions/metrics.json",
    "postfire_soakback": "artifacts/paper_r7_postfire_soakback_audit/metrics.json",
    "sparse_passive_identified_sets": "artifacts/paper_r15_passive_noise_identifiability_boundary/metrics.json",
    "fullfield_transport_capability": "artifacts/paper_r18_fullfield_transport_capability_control/metrics.json",
    "fullfield_five_mechanism_support": "artifacts/paper_r20_cross_scale_consensus_five_mechanisms/metrics.json",
    "chemistry_cracking_parameters": "artifacts/paper_r21_fullfield_nonlinear_parameter_recovery/metrics.json",
    "transport_parameter_boundary": "artifacts/paper_r22_transport_parameter_ratio_recovery/metrics.json",
    "thickness_resolution_boundary": "artifacts/paper_r23_thickness_resolution_boundary/metrics.json",
    "temporal_resolution_boundary": "artifacts/paper_r24_temporal_resolution_boundary/metrics.json",
    "raw_fullfield_noise_stability": "artifacts/paper_r25_fullfield_noise_stability/metrics.json",
    "physics_preserving_denoising_ablation": "artifacts/paper_r26_physics_preserving_denoising_ablation/metrics.json",
    "physics_preserving_noise_validation": "artifacts/paper_r27_physics_preserving_noise_validation/metrics.json",
    "integral_crack_noise_validation": "artifacts/paper_r28_integral_crack_noise_validation/metrics.json",
    "repeated_use_transport_identifiability": "artifacts/paper_r29_repeated_use_transport_identifiability/metrics.json",
    "paired_transport_confirmation": "artifacts/paper_r30_paired_transport_confirmation/metrics.json",
    "unified_noisy_five_mechanism_confirmation": "artifacts/paper_r31_unified_noisy_five_mechanism_confirmation/metrics.json",
    "cross_solver_transport_confirmation": "artifacts/paper_r32_cross_solver_transport_confirmation/metrics.json",
    "anisotropic_transport_open_set_audit": "artifacts/paper_r33_anisotropic_transport_open_set_audit/metrics.json",
    "Darcy_Forchheimer_transport_open_set_audit": "artifacts/paper_r34_darcy_forchheimer_open_set_audit/metrics.json",
    "guarded_transport_decision_confirmation": "artifacts/paper_r35_guarded_transport_decision_confirmation/metrics.json",
    "guarded_five_mechanism_confirmation": "artifacts/paper_r36_guarded_five_mechanism_confirmation/metrics.json",
    "one_percent_guarded_transport_boundary": "artifacts/paper_r37_one_percent_guarded_transport_boundary/metrics.json",
    "directional_transport_library_recovery": "artifacts/paper_r38_directional_transport_library_recovery/metrics.json",
    "cross_scale_directional_transport_confirmation": "artifacts/paper_r39_cross_scale_directional_transport_confirmation/metrics.json",
    "expanded_transport_open_set_guard": "artifacts/paper_r40_expanded_transport_open_set_guard/metrics.json",
    "profiled_Forchheimer_family_failure_audit": "artifacts/paper_r41_profiled_forchheimer_family_recovery/metrics.json",
    "species_profiled_Forchheimer_recovery": "artifacts/paper_r42_species_profiled_forchheimer_recovery/metrics.json",
    "calibrated_diffusivity_Forchheimer_recovery": "artifacts/paper_r43_calibrated_diffusivity_forchheimer_recovery/metrics.json",
    "cross_scale_Forchheimer_boundary": "artifacts/paper_r44_cross_scale_forchheimer_boundary/metrics.json",
    "natural_condition_information_audit": "artifacts/paper_r45_natural_condition_information_audit/metrics.json",
    "excitation_screened_multispecies_recovery": "artifacts/paper_r46_excitation_screened_multispecies_recovery/metrics.json",
    "cross_scale_excitation_screened_confirmation": "artifacts/paper_r47_cross_scale_excitation_screened_confirmation/metrics.json",
    "parameter_denoising_ablation": "artifacts/paper_r48_parameter_denoising_ablation/metrics.json",
    "dual_view_Forchheimer_parameter_recovery": "artifacts/paper_r49_dual_view_forchheimer_parameter_recovery/metrics.json",
    "unified_passive_transport_identification": "artifacts/paper_r50_unified_passive_transport_identification/metrics.json",
    "Forchheimer_strength_generalization": "artifacts/paper_r51_forchheimer_strength_generalization/metrics.json",
    "single_vehicle_condition_budget": "artifacts/paper_r52_single_vehicle_condition_budget/metrics.json",
    "mechanism_to_inverse_health_interface": "artifacts/paper_r53_mechanism_to_inverse_health_interface/metrics.json",
    "inverse_health_conditioning_contract": "artifacts/paper_r53_mechanism_to_inverse_health_interface/conditioning_contract.json",
    "operator_consistent_Forchheimer_topology_development": "artifacts/paper_r54_operator_consistent_forchheimer_topology/development.json",
    "operator_consistent_Forchheimer_topology_failure": "artifacts/paper_r54_operator_consistent_forchheimer_topology/metrics.json",
    "nested_topology_Forchheimer_development": "artifacts/paper_r55_nested_topology_forchheimer_recovery/development.json",
    "nested_topology_Forchheimer_confirmation": "artifacts/paper_r55_nested_topology_forchheimer_recovery/metrics.json",
    "guarded_multiregime_transport_routing_failure": "artifacts/paper_r56_guarded_multiregime_transport_routing/development.json",
    "profiled_directional_Forchheimer_failure": "artifacts/paper_r57_profiled_directional_forchheimer_topology/development.json",
    "strong_nonlinear_topology_set_development": "artifacts/paper_r58_strong_nonlinear_topology_identified_set/development.json",
    "strong_nonlinear_topology_set_confirmation": "artifacts/paper_r58_strong_nonlinear_topology_identified_set/metrics.json",
    "asynchronous_seven_field_observation_validation": "artifacts/paper_r59_asynchronous_seven_field_observation_validation/metrics.json",
    "meridional_resolution_boundary": "artifacts/paper_r60_meridional_resolution_boundary/metrics.json",
    "sparse_asynchronous_seven_field_development": "artifacts/paper_r61_sparse_asynchronous_seven_field_confirmation/development.json",
    "sparse_asynchronous_seven_field_confirmation": "artifacts/paper_r61_sparse_asynchronous_seven_field_confirmation/metrics.json",
    "sparse_observation_noise_boundary_development": "artifacts/paper_r62_sparse_observation_noise_boundary/development.json",
    "sparse_observation_noise_boundary_confirmation": "artifacts/paper_r62_sparse_observation_noise_boundary/metrics.json",
    "minimum_spatial_observation_development": "artifacts/paper_r63_minimum_spatial_observation_confirmation/development.json",
    "minimum_spatial_observation_confirmation": "artifacts/paper_r63_minimum_spatial_observation_confirmation/metrics.json",
    "minimum_spatial_noise_boundary_development": "artifacts/paper_r64_minimum_spatial_noise_boundary/development.json",
    "minimum_spatial_noise_boundary_confirmation": "artifacts/paper_r64_minimum_spatial_noise_boundary/metrics.json",
    "throat_focused_sensor_layout_development": "artifacts/paper_r65_throat_focused_sensor_layout/development.json",
    "throat_focused_sensor_layout_confirmation": "artifacts/paper_r65_throat_focused_sensor_layout/metrics.json",
    "throat_focused_noise_limit_failure": "artifacts/paper_r66_throat_focused_noise_limit/development.json",
    "heldout_bell_geometry_development": "artifacts/paper_r67_heldout_bell_geometry_confirmation/development.json",
    "heldout_bell_geometry_confirmation": "artifacts/paper_r67_heldout_bell_geometry_confirmation/metrics.json",
    "two_sided_bell_geometry_development": "artifacts/paper_r68_two_sided_bell_geometry_confirmation/development.json",
    "two_sided_bell_geometry_confirmation": "artifacts/paper_r68_two_sided_bell_geometry_confirmation/metrics.json",
    "long_bell_nested_sensor_development_failure": "artifacts/paper_r69_long_bell_nested_sensor_augmentation/development.json",
    "seed_aligned_long_bell_sensor_diagnostic": "artifacts/paper_r70_paired_long_bell_sensor_diagnostic/metrics.json",
    "exact_nested_long_bell_sensor_diagnostic": "artifacts/paper_r71_exact_nested_sensor_pairing/metrics.json",
    "long_bell_observation_imperfection_attribution": "artifacts/paper_r72_long_bell_imperfection_attribution/metrics.json",
    "multiscale_sparse_transport_development": "artifacts/paper_r73_multiscale_sparse_transport_development/development.json",
    "multiscale_sparse_transport_confirmation": "artifacts/paper_r74_multiscale_sparse_transport_confirmation/metrics.json",
    "scale_pooled_transport_development_failure": "artifacts/paper_r75_scale_pooled_transport_development/development.json",
    "orthogonal_eiv_transport_development": "artifacts/paper_r76_orthogonal_eiv_transport_development/development.json",
    "orthogonal_eiv_transport_confirmation_failure": "artifacts/paper_r77_orthogonal_eiv_transport_confirmation/metrics.json",
    "candidate_complete_eiv_guard_confirmation": "artifacts/paper_r78_candidate_complete_eiv_guard_confirmation/metrics.json",
    "cross_geometry_guarded_eiv_confirmation": "artifacts/paper_r79_cross_geometry_guarded_eiv_confirmation/metrics.json",
    "three_geometry_guarded_eiv_summary": "artifacts/paper_r80_three_geometry_guarded_eiv_summary/metrics.json",
    "short_bell_eiv_failure_diagnostic": "artifacts/paper_r81_short_bell_eiv_failure_diagnostic/metrics.json",
    "selection_refit_decoupling_development": "artifacts/paper_r82_selection_refit_decoupling_development/development.json",
    "short_bell_decoupled_eiv_confirmation": "artifacts/paper_r83_short_bell_decoupled_eiv_confirmation/metrics.json",
    "corrected_eiv_reference_long_confirmation": "artifacts/paper_r84_corrected_eiv_reference_long_confirmation/metrics.json",
    "corrected_three_geometry_eiv_summary": "artifacts/paper_r85_corrected_three_geometry_summary/metrics.json",
    "transport_coefficient_accuracy_audit": "artifacts/paper_r86_transport_coefficient_accuracy_audit/metrics.json",
    "post_selection_parameter_development": "artifacts/paper_r87_post_selection_parameter_development/development.json",
    "parameter_error_source_attribution": "artifacts/paper_r88_parameter_error_source_attribution/development.json",
    "smooth_spatial_reconstruction_development": "artifacts/paper_r89_physics_smooth_spatial_reconstruction_development/development.json",
    "parameter_spatial_resolution_ladder": "artifacts/paper_r90_parameter_spatial_resolution_ladder/development.json",
    "noise_calibrated_spatial_smoothing_development": "artifacts/paper_r91_noise_calibrated_spatial_smoothing_development/development.json",
    "seven_sensor_parameter_confirmation": "artifacts/paper_r92_seven_sensor_parameter_confirmation/metrics.json",
    "boundary_aware_parameter_development_failure": "artifacts/paper_r93_boundary_aware_parameter_development/development.json",
    "independent_thermal_engineering_benchmark": "artifacts/paper_r94_independent_thermal_engineering_benchmark/metrics.json",
    "layered_csic_surface_qualification": "artifacts/paper_r95_layered_csic_surface_qualification/metrics.json",
    "surface_feedback_diagnostic": "artifacts/paper_r96_surface_feedback_diagnostic/development.json",
    "layered_surface_confirmation": "artifacts/paper_r97_layered_surface_confirmation/metrics.json",
    "engineering_observability_audit": "artifacts/paper_r98_engineering_observability_audit/development.json",
    "passive_cooldown_acoustic_observability": "artifacts/paper_r99_passive_cooldown_acoustic_observability/development.json",
    "domain_shift_truth_qualification_audit": "artifacts/paper_r100_geometry_material_defect_transfer/qualification_diagnostic.json",
    "domain_shift_initial_recovery_boundary": "artifacts/paper_r100_geometry_material_defect_transfer/development.json",
    "bartz_domain_shift_diagnostic": "artifacts/paper_r101_bartz_domain_shift_diagnostic/diagnostic.json",
    "domain_shift_safe_recovery_development": "artifacts/paper_r102_domain_shift_safe_recovery_development/development.json",
    "strict_estimator_agreement_redecision": "artifacts/paper_r103_strict_estimator_agreement_redecision/metrics.json",
    "domain_shift_safe_recovery_confirmation_boundary": "artifacts/paper_r104_domain_shift_safe_recovery_confirmation/metrics.json",
    "cross_scale_union_domain_shift_development": "artifacts/paper_r105_cross_scale_union_domain_shift_development/development.json",
    "cross_scale_union_domain_shift_confirmation": "artifacts/paper_r106_cross_scale_union_domain_shift_confirmation/metrics.json",
    "fullfield_domain_shift_completion_audit": "artifacts/paper_r107_fullfield_domain_shift_completion_audit/metrics.json",
    "axisymmetric_thermoelastic_stress_audit": "artifacts/paper_r108_axisymmetric_thermoelastic_audit/metrics.json",
    "external_fea_contract_preflight": "artifacts/paper_r109_external_fea_contract_preflight/metrics.json",
    "external_full_bell_thermoelastic_audit": "artifacts/paper_r110_external_axisymmetric_fea_audit/metrics.json",
    "orthotropic_compliant_attachment_sensitivity": "artifacts/paper_r111_orthotropic_compliant_attachment_sensitivity/metrics.json",
    "explicit_sliding_flange_archetype": "artifacts/paper_r112_explicit_sliding_flange_archetype/metrics.json",
    "dlr_pressure_fixture_benchmark": "artifacts/paper_r113_dlr_pressure_fixture_benchmark/metrics.json",
    "structural_model_uncertainty_audit": "artifacts/paper_r114_structural_model_uncertainty_audit/metrics.json",
    "constitutively_tied_Forchheimer_development": "artifacts/paper_r115_constitutively_tied_forchheimer_topology/development.json",
    "strong_directional_information_diagnostic": "artifacts/paper_r116_strong_directional_information_diagnostic/diagnostic.json",
    "direction_balanced_weak_windows_development": "artifacts/paper_r117_direction_balanced_weak_windows/development.json",
    "direction_resolved_flux_capability": "artifacts/paper_r118_direction_resolved_flux_capability/development.json",
    "sparse_boundary_flux_information": "artifacts/paper_r119_sparse_boundary_flux_information/development.json",
    "centering_conditioning_audit": "artifacts/paper_r120_centering_conditioning_audit/diagnostic.json",
    "candidate_forward_engineering_observability": "artifacts/paper_r121_candidate_forward_engineering_observability/development.json",
    "engineering_signal_attribution": "artifacts/paper_r122_engineering_signal_attribution/diagnostic.json",
    "full_coupled_transport_truth_qualification": "artifacts/paper_r123_full_coupled_transport_truth/metrics.json",
    "passive_observation_requirement": "artifacts/paper_r124_passive_observation_requirement/diagnostic.json",
    "heldout_nominal_focused_observation": "artifacts/paper_r125_heldout_nominal_focused_observation/preflight.json",
    "focused_observation_transfer": "artifacts/paper_r126_focused_observation_transfer/development.json",
    "material_driver_attribution": "artifacts/paper_r127_material_driver_attribution/development.json",
    "through_conductivity_profile": "artifacts/paper_r128_through_conductivity_profile/development.json",
    "material_profiled_candidate_forward": "artifacts/paper_r129_material_profiled_candidate_forward/development.json",
    "multichannel_material_profiled_adequacy": "artifacts/paper_r130_multichannel_material_profiled_adequacy/development.json",
    "product_precision_contract": "artifacts/paper_r131_product_precision_contract/development.json",
    "correlated_noise_confirmation": "artifacts/paper_r132_correlated_noise_confirmation/confirmation.json",
    "noise_aware_tier_development": "artifacts/paper_r133_noise_aware_tier_development/development.json",
    "covariance_aware_joint_scoring": "artifacts/paper_r134_covariance_aware_joint_scoring/development.json",
    "passive_replication_budget": "artifacts/paper_r135_passive_replication_budget/development.json",
    "conservative_replication_development": "artifacts/paper_r136_conservative_replication_development/development.json",
    "replicated_joint_scoring_confirmation": "artifacts/paper_r137_replicated_joint_scoring_confirmation/confirmation.json",
    "offgrid_material_confirmation": "artifacts/paper_r138_offgrid_material_confirmation/confirmation.json",
    "continuous_material_profile_development": "artifacts/paper_r139_continuous_material_profile_development/development.json",
    "continuous_profile_replication_budget": "artifacts/paper_r140_continuous_profile_replication_budget/development.json",
    "offgrid_continuous_profile_confirmation": "artifacts/paper_r141_offgrid_continuous_profile_confirmation/confirmation.json",
    "independent_gas_operator_preflight": "artifacts/paper_r142_independent_gas_operator_preflight/preflight.json",
    "solver_ensemble_clean_adequacy": "artifacts/paper_r143_solver_ensemble_clean_adequacy/development.json",
    "full_solver_ensemble_development": "artifacts/paper_r144_full_solver_ensemble_development/development.json",
    "independent_offgrid_solver_ensemble_confirmation": "artifacts/paper_r145_independent_offgrid_solver_ensemble_confirmation/confirmation.json",
    "process_variation_clean_transfer": "artifacts/paper_r146_process_variation_clean_transfer/development.json",
    "geometry_conditioned_candidate_forward": "artifacts/paper_r147_geometry_conditioned_candidate_forward/development.json",
    "noisy_process_variation_confirmation": "artifacts/paper_r148_noisy_process_variation_confirmation/confirmation.json",
    "structure_conditional_threshold_development": "artifacts/paper_r149_structure_conditional_threshold_development/development.json",
    "topology_channel_attribution": "artifacts/paper_r150_topology_channel_attribution/development.json",
    "process_robust_32repeat_development": "artifacts/paper_r151_process_robust_replication_development/development.json",
    "conditional_32repeat_development": "artifacts/paper_r152_conditional_32repeat_development/development.json",
    "process_robust_48repeat_development": "artifacts/paper_r153_48repeat_process_robust_development/development.json",
    "conservative_48repeat_development": "artifacts/paper_r154_conservative_48repeat_development/development.json",
    "process_robust_confirmation_near_boundary_failure": "artifacts/paper_r155_process_robust_confirmation/confirmation.json",
    "conditional_48repeat_development_failure": "artifacts/paper_r156_conditional_48repeat_development/development.json",
    "directional_guard_development_failure": "artifacts/paper_r157_directional_guard_development/development.json",
    "physics_weighted_channel_development_failure": "artifacts/paper_r158_physics_weighted_channel_development/development.json",
    "darcy_process_variation_clean_transfer": "artifacts/paper_r159_darcy_process_variation_clean_transfer/development.json",
    "noisy_darcy_process_variation_development": "artifacts/paper_r160_noisy_darcy_process_variation_development/development.json",
    "darcy_process_variation_confirmation": "artifacts/paper_r161_darcy_process_variation_confirmation/confirmation.json",
    "readout_state_drift_clean_gate": "artifacts/paper_r162_readout_state_drift_clean_gate/development.json",
    "noisy_readout_state_drift_development": "artifacts/paper_r163_noisy_readout_state_drift_development/development.json",
    "readout_state_drift_confirmation": "artifacts/paper_r164_readout_state_drift_confirmation/confirmation.json",
    "geometry_metadata_uncertainty_preflight": "artifacts/paper_r165_geometry_metadata_uncertainty_preflight/preflight.json",
    "exact_single_parameter_geometry_attribution": "artifacts/paper_r166_exact_single_parameter_geometry_attribution/attribution.json",
    "expansion_ratio_profile_clean_gate": "artifacts/paper_r167_expansion_ratio_profile_clean_gate/development.json",
    "noisy_expansion_ratio_profile_development": "artifacts/paper_r168_noisy_expansion_ratio_profile_development/development.json",
    "corrected_48repeat_expansion_ratio_profile_development": "artifacts/paper_r168_noisy_expansion_ratio_profile_development/development_48repeat.json",
    "geometry_profile_stratum_audit": "artifacts/paper_r172_geometry_profile_stratum_audit/audit.json",
    "stage1_full_equation_noise_envelope": "artifacts/e6_a_weighted_calibration/metrics.json",
    "internal_ablation": "artifacts/paper_a2_a3_ablation/metrics.json",
    "baseline_independent_weak_equations": "artifacts/paper_b1_independent_weak_baseline/metrics.json",
    "baseline_standard_sindy": "artifacts/paper_b1_standard_sindy/metrics.json",
    "baseline_standard_and_weak_sindy": "artifacts/paper_b2_native_cartesian_baselines/metrics.json",
    "baseline_official_sindy_pi": "artifacts/paper_b3_official_sindypi/metrics.json",
    "runtime_and_closed_loop": "artifacts/paper_b4_runtime_closed_loop/metrics.json",
    "known_multi_mechanism_topology": "artifacts/paper_m4_hierarchical_stable_two_mechanism/metrics.json",
    "multi_mechanism_family": "artifacts/paper_m5_two_mechanism_family_discovery/metrics.json",
    "joint_family_and_topology": "artifacts/paper_m6_joint_multi_family_topology/metrics.json",
    "heterogeneous_three_class_network": "artifacts/paper_h1_heterogeneous_three_class_network/metrics.json",
    "optional_sharing_pooled_no_go": "artifacts/paper_h2_optional_sharing_controls/metrics.json",
    "optional_sharing_partition_stability": "artifacts/paper_h2_partition_stability/metrics.json",
    "joint_family_topology_partition_no_go": "artifacts/paper_h3_joint_family_topology_partition/metrics.json",
    "identifiability_tier_no_go": "artifacts/paper_h3_v6_identifiability_tier/metrics.json",
    "h4c_independent_partition": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4c_v12_noise1_development/hardest-metrics.json",
    "h4c_partial_partition": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4c_v12_noise1_development/partial-metrics.json",
    "h4c_directed_rescue_audit": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4c_directed_rescue_audit_v1.json",
    "h4c_identified_set": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4c_identified_set_v2_noise0_metrics.json",
    "h4_passive_static_separability_no_go": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ae_excitation_design.json",
    "h4_passive_dynamic_separability_no_go": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4af_dynamic_intervention_design.json",
    "h4_local_elasticity_design": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ag_local_elasticity_design.json",
    "h4_response_channel_recovery": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ah_response_channel_recovery.json",
    "h4_response_augmented_adjudication": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ai_response_augmented_adjudication.json",
    "h4_independent_response_bridge": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4aj_independent_response_bridge.json",
    "h4_response_error_budget": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ak_response_error_budget.json",
    "h4_response_component_confirmation": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4al_response_bridge_confirmation.json",
    "h4_spatial_response_bridge": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4am_spatial_response_bridge.json",
    "h4_spatial_response_error_budget": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4an_spatial_response_error_budget.json",
    "h4_spatial_bic_only_confirmation_no_go": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ao_spatial_response_confirmation.json",
    "h4_equivalence_guarded_development": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ap_equivalence_guarded_response.json",
    "h4_equivalence_guarded_confirmation": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4aq_equivalence_guarded_confirmation.json",
    "h4_cross_solver_candidate_out_audit_no_go": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4ar_cross_solver_misspecification_audit.json",
    "h4_model_adequacy_guard_development": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4as_model_adequacy_guard.json",
    "h4_cross_solver_adequacy_confirmation": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4at_cross_solver_adequacy_confirmation.json",
    "h4_expanded_family_recovery_no_go": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4bg_expanded_family_recovery.json",
    "h4_factorial_baseline_design_no_go": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4bh_factorial_baseline_design.json",
    "h4_direct_forward_response_development": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4bi_direct_forward_response.json",
    "h4_open_set_forward_adequacy_no_go": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4bj_open_set_forward_adequacy.json",
    "h4_variance_budget_open_set_development": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4bk_variance_budget_open_set.json",
    "h4_variance_budget_confirmation": "artifacts/paper_h4_complete_mechanism_noise_atlas/h4bl_variance_budget_confirmation.json",
    "h5_identification_health_bridge": "artifacts/paper_h5_identification_to_prognosis/h5a_identification_health_bridge.json",
    "h5_all_family_safe_prognosis": "artifacts/paper_h5_identification_to_prognosis/h5b_all_family_safe_prognosis.json",
    "h5_point_prognosis_confirmation": "artifacts/paper_h5_identification_to_prognosis/h5c_all_family_prognosis_confirmation.json",
    "h5_initial_health_life_intervals": "artifacts/paper_h5_identification_to_prognosis/h5d_calibrated_health_life_intervals.json",
    "h5_marginal_mission_envelope": "artifacts/paper_h5_identification_to_prognosis/h5e_mission_envelope_abstention.json",
    "h5_joint_envelope_audit_no_go": "artifacts/paper_h5_identification_to_prognosis/h5f_joint_envelope_audit.json",
    "h5_joint_support_guard": "artifacts/paper_h5_identification_to_prognosis/h5g_joint_support_guard.json",
    "h5_initial_interval_confirmation_no_go": "artifacts/paper_h5_identification_to_prognosis/h5h_safe_interval_envelope_confirmation.json",
    "h5_seed_blocked_intervals": "artifacts/paper_h5_identification_to_prognosis/h5i_seed_blocked_simultaneous_intervals.json",
    "h5_safe_interval_confirmation": "artifacts/paper_h5_identification_to_prognosis/h5j_safe_prognosis_interval_confirmation.json",
    "network_joint_family_topology": "artifacts/s3b_e4_joint_family_topology/metrics.json",
    "network_reaction_action": "artifacts/s3b_e5b_identifiable_reaction_action/metrics.json",
    "network_weak_coupling_boundary": "artifacts/s3b_e6a_weak_network_boundary/metrics.json",
    "network_excitation_removal": "artifacts/s3b_e6b_excitation_removal/metrics.json",
    "network_sensor_noise_stress": "artifacts/s3b_e6c_sensor_noise_stress/metrics.json",
    "sparse_async_missing": "artifacts/paper_o1_sparse_async_missing/metrics.json",
    "cross_grid_mismatch_no_go": "artifacts/paper_o2_cross_grid_model_mismatch/metrics.json",
    "negative_control_adequacy": "artifacts/paper_o3_negative_control_adequacy/metrics.json",
    "axial_boundary_sensitivity": "artifacts/paper_o4_axial_boundary_sensitivity/metrics.json",
    "autonomous_forecast": "artifacts/s3c_e1_autonomous_forecast/metrics.json",
    "unseen_load_forecast": "artifacts/s3c_e2_unseen_load_programs/metrics.json",
    "repeated_missions": "artifacts/s3c_e3_repeated_mission_cycles/metrics.json",
    "frustum_forecast": "artifacts/s3c_e4_frustum_autonomous_forecast/metrics.json",
    "prediction_intervals": "artifacts/s3c_e5a_calibrated_prediction_intervals/metrics.json",
    "operating_envelope_abstention": "artifacts/s3c_e5b_operating_envelope_abstention/metrics.json",
}
BINARY_ARTIFACTS = {}
for _structure in (
    "Darcy__isotropic",
    "Darcy__directional",
    "Forchheimer__isotropic",
    "Forchheimer__directional",
):
    for _token in ("9p4625", "10p3875", "11p3125", "12p2375"):
        BINARY_ARTIFACTS[f"r138_truth_{_structure}_{_token}"] = (
            "artifacts/paper_r138_offgrid_material_confirmation/"
            f"truth_trajectories/{_structure}/kthrough_{_token}.npz"
        )
    BINARY_ARTIFACTS[f"r142_independent_truth_{_structure}"] = (
        "artifacts/paper_r142_independent_gas_operator_preflight/"
        f"independent_gas_trajectories/{_structure}.npz"
    )
    for _token in ("9p925", "10p850"):
        BINARY_ARTIFACTS[f"r143_independent_candidate_{_structure}_{_token}"] = (
            "artifacts/paper_r143_solver_ensemble_clean_adequacy/"
            f"independent_candidate_trajectories/{_structure}/kthrough_{_token}.npz"
        )
    for _token in ("9p000", "11p775", "12p700"):
        BINARY_ARTIFACTS[f"r144_independent_candidate_{_structure}_{_token}"] = (
            "artifacts/paper_r144_full_solver_ensemble_development/"
            f"independent_candidate_trajectories/{_structure}/kthrough_{_token}.npz"
        )
    for _token in ("9p4625", "11p3125", "12p2375"):
        BINARY_ARTIFACTS[f"r145_independent_truth_{_structure}_{_token}"] = (
            "artifacts/paper_r145_independent_offgrid_solver_ensemble_confirmation/"
            f"independent_offgrid_truth_trajectories/{_structure}/kthrough_{_token}.npz"
        )
    for _token in ("9p000", "9p925", "10p850", "11p775", "12p700"):
        BINARY_ARTIFACTS[f"r147_geometry_independent_{_structure}_{_token}"] = (
            "artifacts/paper_r147_geometry_conditioned_candidate_forward/"
            f"independent_gas_trajectories/{_structure}/kthrough_{_token}.npz"
        )
        if not (_structure.startswith("Forchheimer") and _token == "9p000"):
            BINARY_ARTIFACTS[f"r147_geometry_primary_{_structure}_{_token}"] = (
                "artifacts/paper_r147_geometry_conditioned_candidate_forward/"
                f"primary_trajectories/{_structure}/kthrough_{_token}.npz"
            )
for _case in ("material_batch_shift", "initial_defect_shift", "combined_shift"):
    for _topology in ("isotropic", "directional"):
        BINARY_ARTIFACTS[f"r159_darcy_truth_{_case}_{_topology}"] = (
            "artifacts/paper_r159_darcy_process_variation_clean_transfer/"
            f"truth_trajectories/{_case}/{_topology}.npz"
        )
for _mode in ("length", "wall_thickness", "expansion_ratio", "exit_angle"):
    for _structure in ("Forchheimer__isotropic", "Forchheimer__directional"):
        for _token in ("9p000", "9p925", "10p850", "11p775", "12p700"):
            BINARY_ARTIFACTS[f"r166_primary_{_mode}_{_structure}_{_token}"] = (
                "artifacts/paper_r166_exact_single_parameter_geometry_attribution/"
                f"primary_trajectories/{_mode}/{_structure}/kthrough_{_token}.npz"
            )
            BINARY_ARTIFACTS[f"r166_independent_{_mode}_{_structure}_{_token}"] = (
                "artifacts/paper_r166_exact_single_parameter_geometry_attribution/"
                f"independent_gas_trajectories/{_mode}/{_structure}/kthrough_{_token}.npz"
            )
for _fraction in ("0p25", "0p50", "0p75"):
    for _topology in ("isotropic", "directional"):
        BINARY_ARTIFACTS[f"r167_truth_{_fraction}_{_topology}"] = (
            "artifacts/paper_r167_expansion_ratio_profile_clean_gate/"
            f"truth_trajectories/fraction_{_fraction}_{_topology}.npz"
        )

ENTRY_POINTS = [
    "experiments/paper_r2_geometry_resolved_nozzle_qualification.py",
    "experiments/paper_r3_passive_nozzle_dataset.py",
    "experiments/paper_r5_reality_constrained_natural_conditions.py",
    "experiments/paper_r7_postfire_soakback_audit.py",
    "experiments/paper_r15_passive_noise_identifiability_boundary.py",
    "experiments/paper_r18_fullfield_transport_capability_control.py",
    "experiments/paper_r20_cross_scale_consensus_five_mechanisms.py",
    "experiments/paper_r21_fullfield_nonlinear_parameter_recovery.py",
    "experiments/paper_r22_transport_parameter_ratio_recovery.py",
    "experiments/paper_r23_thickness_resolution_boundary.py",
    "experiments/paper_r24_temporal_resolution_boundary.py",
    "experiments/paper_r25_fullfield_noise_stability.py",
    "experiments/paper_r26_physics_preserving_denoising_ablation.py",
    "experiments/paper_r27_physics_preserving_noise_validation.py",
    "experiments/paper_r28_integral_crack_noise_validation.py",
    "experiments/paper_r29_repeated_use_transport_identifiability.py",
    "experiments/paper_r30_paired_transport_confirmation.py",
    "experiments/paper_r31_unified_noisy_five_mechanism_confirmation.py",
    "experiments/r32_independent_nozzle_gas_solver.py",
    "experiments/paper_r32_cross_solver_transport_confirmation.py",
    "experiments/paper_r33_anisotropic_transport_open_set_audit.py",
    "experiments/paper_r34_darcy_forchheimer_open_set_audit.py",
    "experiments/paper_r35_guarded_transport_decision_confirmation.py",
    "experiments/paper_r36_guarded_five_mechanism_confirmation.py",
    "experiments/paper_r37_one_percent_guarded_transport_boundary.py",
    "experiments/paper_r38_directional_transport_library_recovery.py",
    "experiments/paper_r39_cross_scale_directional_transport_confirmation.py",
    "experiments/paper_r40_expanded_transport_open_set_guard.py",
    "experiments/paper_r41_profiled_forchheimer_family_recovery.py",
    "experiments/paper_r42_species_profiled_forchheimer_recovery.py",
    "experiments/paper_r43_calibrated_diffusivity_forchheimer_recovery.py",
    "experiments/paper_r44_cross_scale_forchheimer_boundary.py",
    "experiments/paper_r45_natural_condition_information_audit.py",
    "experiments/paper_r46_excitation_screened_multispecies_recovery.py",
    "experiments/paper_r47_cross_scale_excitation_screened_confirmation.py",
    "experiments/paper_r48_parameter_denoising_ablation.py",
    "experiments/paper_r49_dual_view_forchheimer_parameter_recovery.py",
    "experiments/paper_r50_unified_passive_transport_identification.py",
    "experiments/paper_r51_forchheimer_strength_generalization.py",
    "experiments/paper_r52_single_vehicle_condition_budget.py",
    "experiments/paper_r53_mechanism_to_inverse_health_interface.py",
    "experiments/paper_r54_operator_consistent_forchheimer_topology.py",
    "experiments/paper_r55_nested_topology_forchheimer_recovery.py",
    "experiments/paper_r56_guarded_multiregime_transport_routing.py",
    "experiments/paper_r57_profiled_directional_forchheimer_topology.py",
    "experiments/paper_r58_strong_nonlinear_topology_identified_set.py",
    "experiments/paper_r59_asynchronous_seven_field_observation_validation.py",
    "experiments/paper_r60_meridional_resolution_boundary.py",
    "experiments/paper_r61_sparse_asynchronous_seven_field_confirmation.py",
    "experiments/paper_r62_sparse_observation_noise_boundary.py",
    "experiments/paper_r63_minimum_spatial_observation_confirmation.py",
    "experiments/paper_r64_minimum_spatial_noise_boundary.py",
    "experiments/paper_r65_throat_focused_sensor_layout.py",
    "experiments/paper_r66_throat_focused_noise_limit.py",
    "experiments/paper_r67_heldout_bell_geometry_confirmation.py",
    "experiments/paper_r68_two_sided_bell_geometry_confirmation.py",
    "experiments/paper_r69_long_bell_nested_sensor_augmentation.py",
    "experiments/paper_r70_paired_long_bell_sensor_diagnostic.py",
    "experiments/paper_r71_exact_nested_sensor_pairing.py",
    "experiments/paper_r72_long_bell_imperfection_attribution.py",
    "experiments/paper_r73_multiscale_sparse_transport_development.py",
    "experiments/paper_r74_multiscale_sparse_transport_confirmation.py",
    "experiments/paper_r75_scale_pooled_transport_development.py",
    "experiments/paper_r76_orthogonal_eiv_transport_development.py",
    "experiments/paper_r77_orthogonal_eiv_transport_confirmation.py",
    "experiments/paper_r78_candidate_complete_eiv_guard_confirmation.py",
    "experiments/paper_r79_cross_geometry_guarded_eiv_confirmation.py",
    "experiments/paper_r80_three_geometry_guarded_eiv_summary.py",
    "experiments/paper_r81_short_bell_eiv_failure_diagnostic.py",
    "experiments/paper_r82_selection_refit_decoupling_development.py",
    "experiments/paper_r83_short_bell_decoupled_eiv_confirmation.py",
    "experiments/paper_r84_corrected_eiv_reference_long_confirmation.py",
    "experiments/paper_r85_corrected_three_geometry_summary.py",
    "experiments/paper_r86_transport_coefficient_accuracy_audit.py",
    "experiments/paper_r87_post_selection_parameter_development.py",
    "experiments/paper_r88_parameter_error_source_attribution.py",
    "experiments/paper_r89_physics_smooth_spatial_reconstruction_development.py",
    "experiments/paper_r90_parameter_spatial_resolution_ladder.py",
    "experiments/paper_r91_noise_calibrated_spatial_smoothing_development.py",
    "experiments/paper_r92_seven_sensor_parameter_confirmation.py",
    "experiments/paper_r93_boundary_aware_parameter_development.py",
    "experiments/paper_r94_independent_thermal_engineering_benchmark.py",
    "experiments/paper_r95_layered_csic_surface_qualification.py",
    "experiments/paper_r96_surface_feedback_diagnostic.py",
    "experiments/paper_r97_layered_surface_confirmation.py",
    "experiments/paper_r98_engineering_observability_audit.py",
    "experiments/paper_r99_passive_cooldown_acoustic_observability.py",
    "experiments/paper_r100_geometry_material_defect_transfer.py",
    "experiments/paper_r101_bartz_domain_shift_diagnostic.py",
    "experiments/paper_r102_domain_shift_safe_recovery_development.py",
    "experiments/paper_r103_strict_estimator_agreement_redecision.py",
    "experiments/paper_r104_domain_shift_safe_recovery_confirmation.py",
    "experiments/paper_r105_cross_scale_union_domain_shift_development.py",
    "experiments/paper_r106_cross_scale_union_domain_shift_confirmation.py",
    "experiments/paper_r107_fullfield_domain_shift_completion_audit.py",
    "experiments/paper_r108_axisymmetric_thermoelastic_audit.py",
    "experiments/paper_r109_external_fea_contract_preflight.py",
    "experiments/paper_r110_external_axisymmetric_fea_audit.py",
    "experiments/paper_r111_orthotropic_compliant_attachment_sensitivity.py",
    "experiments/paper_r112_explicit_sliding_flange_archetype.py",
    "experiments/paper_r113_dlr_pressure_fixture_benchmark.py",
    "experiments/paper_r114_structural_model_uncertainty_audit.py",
    "experiments/paper_r115_constitutively_tied_forchheimer_topology.py",
    "experiments/paper_r116_strong_directional_information_diagnostic.py",
    "experiments/paper_r117_direction_balanced_weak_windows.py",
    "experiments/paper_r118_direction_resolved_flux_capability.py",
    "experiments/paper_r119_sparse_boundary_flux_information.py",
    "experiments/paper_r120_centering_conditioning_audit.py",
    "experiments/paper_r121_candidate_forward_engineering_observability.py",
    "experiments/paper_r122_engineering_signal_attribution.py",
    "experiments/paper_r123_full_coupled_transport_truth.py",
    "experiments/paper_r124_passive_observation_requirement.py",
    "experiments/paper_r125_heldout_nominal_focused_observation.py",
    "experiments/paper_r126_focused_observation_transfer.py",
    "experiments/paper_r127_material_driver_attribution.py",
    "experiments/paper_r128_through_conductivity_profile.py",
    "experiments/paper_r129_material_profiled_candidate_forward.py",
    "experiments/paper_r130_multichannel_material_profiled_adequacy.py",
    "experiments/paper_r131_product_precision_contract.py",
    "experiments/paper_r132_correlated_noise_confirmation.py",
    "experiments/paper_r133_noise_aware_tier_development.py",
    "experiments/paper_r134_covariance_aware_joint_scoring.py",
    "experiments/paper_r135_passive_replication_budget.py",
    "experiments/paper_r136_conservative_replication_development.py",
    "experiments/paper_r137_replicated_joint_scoring_confirmation.py",
    "experiments/paper_r138_offgrid_material_confirmation.py",
    "experiments/paper_r139_continuous_material_profile_development.py",
    "experiments/paper_r140_continuous_profile_replication_budget.py",
    "experiments/paper_r141_offgrid_continuous_profile_confirmation.py",
    "experiments/paper_r142_independent_gas_operator_preflight.py",
    "experiments/paper_r143_solver_ensemble_clean_adequacy.py",
    "experiments/paper_r144_full_solver_ensemble_development.py",
    "experiments/paper_r145_independent_offgrid_solver_ensemble_confirmation.py",
    "experiments/paper_r146_process_variation_clean_transfer.py",
    "experiments/paper_r147_geometry_conditioned_candidate_forward.py",
    "experiments/paper_r148_noisy_process_variation_confirmation.py",
    "experiments/paper_r149_structure_conditional_threshold_development.py",
    "experiments/paper_r150_topology_channel_attribution.py",
    "experiments/paper_r151_process_robust_replication_development.py",
    "experiments/paper_r152_conditional_32repeat_development.py",
    "experiments/paper_r153_48repeat_process_robust_development.py",
    "experiments/paper_r154_conservative_48repeat_development.py",
    "experiments/paper_r155_process_robust_confirmation.py",
    "experiments/paper_r156_conditional_48repeat_development.py",
    "experiments/paper_r157_directional_guard_development.py",
    "experiments/paper_r158_physics_weighted_channel_development.py",
    "experiments/paper_r159_darcy_process_variation_clean_transfer.py",
    "experiments/paper_r160_noisy_darcy_process_variation_development.py",
    "experiments/paper_r161_darcy_process_variation_confirmation.py",
    "experiments/paper_r162_readout_state_drift_clean_gate.py",
    "experiments/paper_r163_noisy_readout_state_drift_development.py",
    "experiments/paper_r164_readout_state_drift_confirmation.py",
    "experiments/paper_r165_geometry_metadata_uncertainty_preflight.py",
    "experiments/paper_r166_exact_single_parameter_geometry_attribution.py",
    "experiments/paper_r167_expansion_ratio_profile_clean_gate.py",
    "experiments/paper_r168_noisy_expansion_ratio_profile_development.py",
    "experiments/paper_r169_96repeat_geometry_profile_budget.py",
    "experiments/paper_r170_geometry_profile_confirmation.py",
    "experiments/paper_r171_full_family_geometry_profile_clean_gate.py",
    "experiments/paper_r172_geometry_profile_stratum_audit.py",
    "experiments/e6_a_weighted_calibration.py",
    "experiments/paper_a2_a3_ablation.py",
    "experiments/paper_b1_independent_weak_baseline.py",
    "experiments/paper_b1_standard_sindy.py",
    "experiments/paper_b2_native_cartesian_baselines.py",
    "experiments/paper_b3_official_sindypi.py",
    "experiments/paper_b4_runtime_closed_loop.py",
    "experiments/paper_m4_hierarchical_stable_two_mechanism.py",
    "experiments/paper_m5_two_mechanism_family_discovery.py",
    "experiments/paper_m6_joint_multi_family_topology.py",
    "experiments/paper_h1_heterogeneous_three_class_network.py",
    "experiments/paper_h2_optional_sharing_controls.py",
    "experiments/paper_h2_partition_stability.py",
    "experiments/paper_h3_joint_family_topology_partition.py",
    "experiments/paper_h3_v6_identifiability_tier.py",
    "experiments/paper_h4_complete_mechanism_noise_atlas.py",
    "experiments/paper_h4ae_excitation_design.py",
    "experiments/paper_h4af_dynamic_intervention_design.py",
    "experiments/paper_h4ag_local_elasticity_design.py",
    "experiments/paper_h4ah_response_channel_recovery.py",
    "experiments/h4aj_independent_response_source.py",
    "experiments/paper_h4aj_independent_response_bridge.py",
    "experiments/paper_h4ak_response_error_budget.py",
    "experiments/paper_h4al_response_bridge_confirmation.py",
    "experiments/h4am_independent_spatial_response_source.py",
    "experiments/paper_h4am_spatial_response_bridge.py",
    "experiments/paper_h4an_spatial_response_error_budget.py",
    "experiments/paper_h4ao_spatial_response_confirmation.py",
    "experiments/paper_h4ap_equivalence_guarded_response.py",
    "experiments/paper_h4aq_equivalence_guarded_confirmation.py",
    "experiments/h4ar_independent_finite_volume_response_source.py",
    "experiments/paper_h4ar_cross_solver_misspecification_audit.py",
    "experiments/paper_h4as_model_adequacy_guard.py",
    "experiments/paper_h4at_cross_solver_adequacy_confirmation.py",
    "experiments/paper_h4bg_expanded_family_recovery.py",
    "experiments/paper_h4bh_factorial_baseline_design.py",
    "experiments/paper_h4bi_direct_forward_response.py",
    "experiments/paper_h4bj_open_set_forward_adequacy.py",
    "experiments/paper_h4bk_variance_budget_open_set.py",
    "experiments/paper_h4bl_variance_budget_confirmation.py",
    "experiments/paper_h5a_identification_health_bridge.py",
    "experiments/paper_h5b_all_family_safe_prognosis.py",
    "experiments/paper_h5c_all_family_prognosis_confirmation.py",
    "experiments/paper_h5d_calibrated_health_life_intervals.py",
    "experiments/paper_h5e_mission_envelope_abstention.py",
    "experiments/paper_h5f_joint_envelope_audit.py",
    "experiments/paper_h5g_joint_support_guard.py",
    "experiments/paper_h5h_safe_interval_envelope_confirmation.py",
    "experiments/paper_h5i_seed_blocked_simultaneous_intervals.py",
    "experiments/paper_h5j_safe_prognosis_interval_confirmation.py",
    "experiments/paper_h4c_final_union.py",
    "experiments/paper_h4c_identified_set.py",
    "experiments/paper_h4c_noise1_hardest_coverage.py",
    "experiments/paper_h4c_noise1_partial_coverage.py",
    "experiments/s3b_e4_joint_family_topology.py",
    "experiments/s3b_e5b_identifiable_reaction_action.py",
    "experiments/s3b_e6a_weak_network_boundary.py",
    "experiments/s3b_e6b_excitation_removal.py",
    "experiments/s3b_e6c_sensor_noise_stress.py",
    "experiments/paper_o1_sparse_async_missing.py",
    "experiments/paper_o2_cross_grid_model_mismatch.py",
    "experiments/paper_o3_negative_control_adequacy.py",
    "experiments/paper_o4_axial_boundary_sensitivity.py",
    "experiments/s3c_e1_autonomous_forecast.py",
    "experiments/s3c_e2_unseen_load_programs.py",
    "experiments/s3c_e3_repeated_mission_cycles.py",
    "experiments/s3c_e4_frustum_autonomous_forecast.py",
    "experiments/s3c_e5a_calibrated_prediction_intervals.py",
    "experiments/s3c_e5b_operating_envelope_abstention.py",
    "experiments/build_paper_assets.py",
]

SUPPORTING_CODE = [
    "reproducibility/run_experiment_background.ps1",
    "reproducibility/background_experiment_worker.ps1",
    "src/wphid/discovery/conditioning.py",
    "src/wphid/truth/reusable_nozzle.py",
    "src/wphid/truth/csic_surface.py",
    "src/wphid/truth/axisymmetric_thermoelastic.py",
    "src/wphid/validation/external_fea.py",
    "src/wphid/engineering_observations.py",
    "experiments/build_nozzle_truth_architecture_figure.py",
    "experiments/build_domain_shift_evidence_figure.py",
]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def build(output=None):
    artifact_rows = []
    for role, relative in ARTIFACTS.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate", {})
        artifact_rows.append({
            "role": role, "path": relative.replace("\\", "/"),
            "bytes": path.stat().st_size, "sha256": _sha256(path),
            "protocol_sha256": (
                payload.get("protocol_sha256")
                or payload.get("parent_r3_protocol_sha256")
            ),
            "decision": aggregate.get("decision"),
            "gate_pass": aggregate.get("gate_pass"),
        })
    for role, relative in BINARY_ARTIFACTS.items():
        path = ROOT / relative
        artifact_rows.append({
            "role": role,
            "path": relative.replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "media_type": "application/x-npz",
            "protocol_sha256": None,
            "decision": None,
            "gate_pass": None,
        })
    code_rows = []
    for relative in ENTRY_POINTS:
        path = ROOT / relative
        code_rows.append({"path": relative, "sha256": _sha256(path)})
    supporting_code_rows = []
    for relative in SUPPORTING_CODE:
        path = ROOT / relative
        supporting_code_rows.append({"path": relative, "sha256": _sha256(path)})
    packages = {}
    for package in ("numpy", "scipy", "pytest", "scikit-learn", "pysindy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    status = _git("status", "--porcelain")
    manifest = {
        "schema": "WPHID_PAPER_REPRODUCIBILITY_MANIFEST_V1",
        "git": {"commit": _git("rev-parse", "HEAD"),
                "branch": _git("branch", "--show-current"),
                "worktree_clean": not bool(status),
                "dirty_entry_count": len(status.splitlines())},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(), "packages": packages},
        "verification": json.loads((ROOT / "reproducibility/test_verification.json")
                                   .read_text(encoding="utf-8")),
        "artifacts": artifact_rows,
        "entry_points": code_rows,
        "supporting_code": supporting_code_rows,
        "paper_assets": {
            "path": "docs/paper/paper_asset_manifest.json",
            "sha256": _sha256(ROOT / "docs/paper/paper_asset_manifest.json"),
        },
    }
    destination = ROOT / (output or "reproducibility/canonical_manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps({"artifact_count": len(result["artifacts"]),
                      "git": result["git"]}, indent=2))
