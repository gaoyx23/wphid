import experiments.s3c_e5b_operating_envelope_abstention as experiment


def test_s3c_e5b_admits_and_abstains_for_explicit_reasons():
    admitted = experiment.forecast_decision(
        cycles=4, geometry="nominal", coupling_stability=1., peak_load=1.15)
    rejected = experiment.forecast_decision(
        cycles=5, geometry="unknown", coupling_stability=.5, peak_load=2.5)
    assert admitted["decision"] == "ADMIT"
    assert rejected["decision"] == "ABSTAIN"
    assert len(rejected["reasons"]) == 4
