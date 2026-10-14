from experiments.paper_o3_negative_control_adequacy import PROTOCOL, _admitted


def test_negative_control_gate_has_frozen_accept_and_abstain_panels():
    assert PROTOCOL["maximum_false_admission_rate"] == 0.2
    assert {panel["expected"] for panel in PROTOCOL["panels"].values()} == {
        "accept", "abstain"
    }


def test_false_admission_uses_only_qualified_output():
    assert not _admitted({"qualified_reaction_family": None,
                          "qualified_leakage_family": None})
    assert _admitted({"qualified_reaction_family": "candidate",
                      "qualified_leakage_family": None})
