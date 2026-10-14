import experiments.s3c_e2_unseen_load_programs as experiment


def test_s3c_e2_unseen_program_is_autonomous():
    row = experiment.run_case("dual_pulse", 34201)
    assert set(row["nrmse_by_horizon"]["1.4"]) == set(experiment.FIELDS)
    assert row["maximum_final_horizon_nrmse"] == max(
        row["nrmse_by_horizon"]["1.4"].values())
