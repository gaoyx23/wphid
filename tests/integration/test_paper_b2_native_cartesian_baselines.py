import experiments.paper_b2_native_cartesian_baselines as experiment


def test_native_cartesian_official_libraries_share_data_and_truth_support():
    x, t, trajectories = experiment._observations(35101, True)
    assert trajectories[0].shape == (x.size, t.size, len(experiment.FIELDS))
    for method in ("standard_sindy", "weak_sindy"):
        model = experiment._model(method, x, t, .1, 35101)
        assert model.optimizer.__class__.__name__ == "STLSQ"
