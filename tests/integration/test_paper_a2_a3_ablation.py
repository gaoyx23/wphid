from experiments import paper_a2_a3_ablation as experiment


def test_matched_component_ablation_exposes_false_claims(tmp_path):
    metrics = experiment.run_experiment(tmp_path / "metrics.json")
    a2 = metrics["a2_no_condition_stability"]
    a3 = metrics["a3_no_output_tiering"]
    assert a2["full_null_false_admission_rate"] == 0.0
    assert a2["no_stability_null_false_admission_rate"] == 0.6
    assert a3["tiered_correct_output_rate"] == 1.0
    assert a3["forced_full_decomposition_wrong_claim_rate"] == 0.4
    assert metrics["aggregate"]["decision"] == "GO"
