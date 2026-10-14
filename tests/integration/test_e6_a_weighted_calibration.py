from experiments.e6_a_weighted_calibration import run_experiment


def test_e6_a_v3_separates_output_and_coverage(tmp_path):
    metrics = run_experiment(
        tmp_path / "metrics.json", multipliers=[2.0],
        reliability_seeds=[12001, 12002],
        conformal_seeds=[12021, 12022],
        confirmation_seeds=[12201],
    )
    curve = metrics["curves"][0]
    assert 0.0 <= curve["estimator_output_rate"] <= 1.0
    assert 0.0 <= curve["conditional_joint_coverage"] <= 1.0
    assert 0.0 <= curve["effective_joint_coverage"] <= 1.0
    labels = {row["evaluation_label"] for row in metrics["confirmation"]}
    assert labels <= {"correct_output", "wrong_output", "non-identifiable"}
