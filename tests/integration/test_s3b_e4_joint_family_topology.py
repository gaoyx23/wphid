from experiments.s3b_e4_joint_family_topology import PROTOCOL


def test_joint_search_refines_multiple_screened_hypotheses():
    assert PROTOCOL["refinement_count"] >= 2
    assert PROTOCOL["refinement_count"] < 35
