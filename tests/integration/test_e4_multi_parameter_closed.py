from experiments.e4_multi_parameter_closed import run_experiment


def test_e4_uses_weak_reconstruction_and_passes_confirmation(tmp_path):
    metrics = run_experiment(tmp_path / "metrics.json")
    assert metrics["aggregate"]["gate_pass"]
    assert metrics["aggregate"]["confirmation_success_rate"] >= 0.8
    assert all(row["stable_extrapolation"] for row in metrics["confirmation"])
    expected = (len(metrics["protocol"]["simex_levels"])
                * metrics["protocol"]["simex_replicates"])
    assert all(len(row["reconstruction_audit"]) == expected
               for row in metrics["confirmation"])
