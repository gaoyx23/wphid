from experiments import paper_b4_runtime_closed_loop as experiment


def test_standard_closed_loop_case_is_retained_on_success_or_failure():
    coefficients = experiment._standard_coefficients_by_seed()[31901]
    row = experiment.run_standard_forecast("nominal", 31901, 34101, coefficients)
    assert row["method"] == "official_pointwise_sindy"
    assert row["integration_success"] in (True, False)
    assert "maximum_nrmse" in row
