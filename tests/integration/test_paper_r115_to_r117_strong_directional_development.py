import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r115_constitutive_tie_improves_isotropic_but_fails_directional_control():
    payload = _load(
        "artifacts/paper_r115_constitutively_tied_forchheimer_topology/development.json"
    )
    assert payload["development_gate_passed"] is False
    assert payload["all_trajectories_reused"] is True
    assert payload["aggregate"]["forchheimer_strong_isotropic"]["truth_coverage_count"] == 3
    assert payload["aggregate"]["forchheimer_strong_isotropic"]["mean_reported_set_width"] == 2.0
    assert payload["aggregate"]["forchheimer_strong_directional"]["truth_coverage_count"] == 0
    assert payload["aggregate"]["forchheimer_strong_directional"]["wrong_only_count"] == 3
    assert not (
        ROOT / "artifacts/paper_r115_constitutively_tied_forchheimer_topology/metrics.json"
    ).exists()


def test_r116_attributes_failure_before_measurement_noise():
    payload = _load(
        "artifacts/paper_r116_strong_directional_information_diagnostic/diagnostic.json"
    )
    diagnosis = payload["diagnosis"]
    assert payload["new_trajectory_runs"] == 0
    assert payload["confirmation_seeds_used"] is False
    assert diagnosis["directional_truth_covered_at_both_clean_scales"] is False
    assert diagnosis["maximum_flux_column_cosine"] > 0.9999
    assert diagnosis["minimum_raw_field_separation_to_noise_ratio"] > 1.0
    assert diagnosis["r115_confirmation_remains_blocked"] is True


def test_r116_records_species_specific_directional_information_imbalance():
    raw = _load(
        "artifacts/paper_r116_strong_directional_information_diagnostic/diagnostic.json"
    )["raw_field_information"]
    assert raw["pO2"]["meridional_to_thickness_gradient_energy_ratio"] < 0.01
    assert raw["pH2O"]["meridional_to_thickness_gradient_energy_ratio"] < 0.10
    assert raw["pH2O"]["field_separation_to_noise_ratio"] > 100.0


def test_r117_balanced_windows_do_not_unlock_confirmation():
    payload = _load(
        "artifacts/paper_r117_direction_balanced_weak_windows/development.json"
    )
    assert payload["development_gate_passed"] is False
    assert payload["confirmation_seeds_used"] is False
    assert payload["predeclared_method"]["same_mask_for_every_candidate"] is True
    assert payload["aggregate"]["strong_isotropic"]["truth_coverage_count"] == 3
    assert payload["aggregate"]["strong_directional"]["truth_coverage_count"] == 0
    assert payload["aggregate"]["strong_directional"]["wrong_only_count"] == 3


def test_r118_direction_resolved_flux_is_an_upper_bound_not_a_sensor_claim():
    payload = _load(
        "artifacts/paper_r118_direction_resolved_flux_capability/development.json"
    )
    assert payload["capability_gate_passed"] is True
    assert payload["all_state_trajectories_reused"] is True
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["confirmation_seeds_used"] is False
    assert payload["observation_contract"]["full_field_flux"] is True
    assert "does not establish" in payload["claim_boundary"]


def test_r118_resolves_both_strong_nonlinear_topologies_in_development():
    payload = _load(
        "artifacts/paper_r118_direction_resolved_flux_capability/development.json"
    )
    for panel in ("strong_isotropic", "strong_directional"):
        aggregate = payload["aggregate"][panel]
        assert aggregate["case_count"] == 3
        assert aggregate["truth_coverage_count"] == 3
        assert aggregate["unique_truth_count"] == 3
        assert aggregate["wrong_only_count"] == 0
        assert aggregate["mean_set_width"] == 1.0


def test_r119_outer_leakage_alone_cannot_identify_directional_truth():
    payload = _load(
        "artifacts/paper_r119_sparse_boundary_flux_information/development.json"
    )
    assert payload["all_state_trajectories_reused"] is True
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["confirmation_seeds_used"] is False
    for noise in ("1pct", "3pct", "5pct"):
        block = payload["aggregate"]["outer_normal_leakage_only"][noise]
        assert block["strong_isotropic"]["truth_coverage_count"] == 3
        assert block["strong_directional"]["truth_coverage_count"] == 0
        assert block["strong_directional"]["wrong_only_count"] == 3


def test_r119_sparse_direction_pair_preserves_information_through_five_percent_noise():
    payload = _load(
        "artifacts/paper_r119_sparse_boundary_flux_information/development.json"
    )
    assert payload["sparse_direction_pair_information_sufficient"] is True
    assert "not treated as an already available" in payload["claim_boundary"]
    for noise in ("1pct", "3pct", "5pct"):
        for panel in ("strong_isotropic", "strong_directional"):
            block = payload["aggregate"]["sparse_boundary_direction_pair"][noise][panel]
            assert block["truth_coverage_count"] == 3
            assert block["unique_truth_count"] == 3
            assert block["wrong_only_count"] == 0


def test_r120_centering_does_not_hide_the_operator_alias():
    payload = _load(
        "artifacts/paper_r120_centering_conditioning_audit/diagnostic.json"
    )
    assert payload["new_forward_trajectory_runs"] == 0
    assert payload["confirmation_seeds_used"] is False
    assert payload["summary"]["centering_promoted_to_method"] is False
    assert (
        payload["summary"][
            "directional_truth_covered_after_within_condition_centering_scale_count"
        ]
        == 0
    )
    assert payload["summary"]["maximum_centered_topology_column_cosine"] > 0.9999
    for scale in ("intermediate", "global"):
        isotropic = payload["results"]["strong_isotropic"][scale]
        directional = payload["results"]["strong_directional"][scale]
        assert isotropic["within_condition_centered_truth_covered"] is True
        assert directional["within_condition_centered_truth_covered"] is False
