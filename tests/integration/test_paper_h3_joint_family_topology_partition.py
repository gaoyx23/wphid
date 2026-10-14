from experiments import paper_h3_joint_family_topology_partition as experiment


def test_partition_enumeration_matches_bell_counts():
    assert len(experiment._partitions((True, False, False))) == 1
    assert len(experiment._partitions((True, True, False))) == 2
    assert len(experiment._partitions((True, True, True))) == 5


def test_truth_roles_cover_required_controls():
    assert set(experiment.TRUTH_CASES) == {
        "shared", "partial", "independent", "near_equivalent", "null"
    }
    assert experiment.TRUTH_CASES["partial"]["partition"] == (0, 0, 1)
    assert experiment.TRUTH_CASES["partial"]["expected_output"] == "TO|D"
    assert experiment.TRUTH_CASES["near_equivalent"]["expected_output"] == "abstain"


def test_protocol_opens_family_topology_and_partition():
    assert len(experiment.FAMILIES) >= 4
    assert len(experiment.OPEN_TOPOLOGIES) == 7
    assert len(experiment._partitions((True, True, True))) == 5
    assert experiment.PROTOCOL_HASH
