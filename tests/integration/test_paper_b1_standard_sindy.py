import experiments.paper_b1_standard_sindy as experiment


def test_pointwise_sindy_uses_official_optimizer_and_fixed_library():
    assert experiment.ps.__version__ == "2.1.0"
    matrices, targets = experiment._pointwise_design(31801, True)
    assert len(matrices) == len(experiment.FIELDS)
    assert all(matrix.shape[1] == len(experiment.FEATURE_NAMES) for matrix in matrices)
    assert all(matrix.shape[0] == target.size for matrix, target in zip(matrices, targets))
