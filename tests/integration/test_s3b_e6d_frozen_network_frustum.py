import experiments.s3b_e6d_frozen_network_frustum as experiment


def test_s3b_e6d_protocol_and_single_case():
    assert set(experiment.PROTOCOL["geometries"]) == {"nominal", "steep", "thin_tapered"}
    row = experiment.run_case("nominal", 33101)
    assert set(row["excess_normalized_residuals"]) == set(experiment.FIELDS)
    assert row["maximum_network_excess_residual"] == max(
        row["excess_normalized_residuals"].values())
    assert row["maximum_relative_operator_action_error"] == max(
        row["relative_operator_action_errors"].values())
