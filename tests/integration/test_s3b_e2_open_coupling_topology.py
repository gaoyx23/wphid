from experiments.s3b_e2_open_coupling_topology import (
    NULL_TOPOLOGY,
    TOPOLOGIES,
    _topology_name,
)


def test_candidate_topologies_are_complete_and_unique():
    assert len(TOPOLOGIES) == 8
    assert len(set(TOPOLOGIES)) == 8
    assert TOPOLOGIES[0] == NULL_TOPOLOGY
    assert _topology_name(NULL_TOPOLOGY) == "null"
