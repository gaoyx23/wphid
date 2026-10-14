import experiments.s3c_e5a_calibrated_prediction_intervals as experiment


def test_s3c_e5a_uses_disjoint_calibration_and_evaluation_seeds():
    assert set(experiment.PROTOCOL["calibration_seeds"]).isdisjoint(
        experiment.PROTOCOL["evaluation_seeds"])
    assert len(experiment._ensemble()) == 5
