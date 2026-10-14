from experiments.s3b_e6c_sensor_noise_stress import SCENARIOS


def test_sensor_noise_stress_contains_iid_heterogeneous_and_correlated_cases():
    assert SCENARIOS["baseline_iid"]["rho"] == 0.0
    assert SCENARIOS["heterogeneous"]["noise_fractions"]["damage"] > .0005
    assert SCENARIOS["strong_ar1"]["rho"] > SCENARIOS["moderate_ar1"]["rho"]
