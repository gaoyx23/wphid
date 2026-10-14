from experiments.s3b_e5b_identifiable_reaction_action import PROTOCOL


def test_action_audit_separates_action_and_decomposition_thresholds():
    assert PROTOCOL["maximum_reaction_action_error"] < 1.0
    assert PROTOCOL["noise_levels"] == sorted(PROTOCOL["noise_levels"])
