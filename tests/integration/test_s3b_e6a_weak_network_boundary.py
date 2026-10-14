from experiments.s3b_e6a_weak_network_boundary import PROTOCOL


def test_weak_network_boundary_includes_nominal_and_zero_controls():
    assert PROTOCOL["reaction_scales"][0] == 1.0
    assert PROTOCOL["reaction_scales"][-1] == 0.0
