import experiments.s3c_e1_autonomous_forecast as experiment


def test_s3c_e1_is_closed_loop_after_initial_state():
    row = experiment.run_case("nominal", 34101)
    assert set(row["nrmse_by_horizon"]["1.0"]) == set(experiment.FIELDS)
    assert row["maximum_final_horizon_nrmse"] == max(
        row["nrmse_by_horizon"]["1.0"].values())
