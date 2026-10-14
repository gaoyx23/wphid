import hashlib,json
from experiments.run_nested_passive_data_pilot import ROOT


def test_fixed_packet_hashes_are_reused_and_holdout_excluded():
    base=ROOT/"artifacts/_audits/nested_passive_data_pilot"
    result=json.loads((base/"audit.json").read_text(encoding="utf-8"))
    assert result["complete"] and result["new_physical_trajectories"]==0
    assert result["new_observation_packets"]==10
    for level in result["levels"].values():
        assert not set(level["tasks"]) & set(result["protocol"]["holdout"])
        for name,digest in level["input_packet_sha256"].items():
            assert digest==result["observations"][name]
            assert digest==hashlib.sha256((base/"observations"/f"{name}.npz").read_bytes()).hexdigest()


def test_negative_pilot_not_promoted_to_confirmation():
    result=json.loads((ROOT/"artifacts/_audits/nested_passive_data_pilot/audit.json").read_text(encoding="utf-8"))
    assert not result["development_truth_retained_all_levels"]
    assert result["levels"]["eight_tasks"]["scoring_only"]["all_five_truth_covered"]
    assert result["protocol"]["no_automatic_followup_noise_panel"]
    assert all(row["selection_unchanged_after_validation"] for row in result["holdout_validation"].values())


def test_no_refit_diagnostic_binds_original_results():
    raw=(ROOT/"artifacts/_audits/nested_passive_data_pilot/audit.json").read_bytes()
    result=json.loads((ROOT/"artifacts/_audits/nested_passive_saved_diagnostic/audit.json").read_text(encoding="utf-8"))
    assert result["source_sha256"]==hashlib.sha256(raw).hexdigest()
    assert result["new_model_fits"]==result["new_noise_draws"]==result["new_simulations"]==0
