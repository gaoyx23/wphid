from experiments.s3b_e6b_excitation_removal import SCENARIOS


def test_each_removal_keeps_three_distinct_conditions():
    assert len(SCENARIOS["all_conditions"]) == 4
    assert all(len(value) == 3 for name, value in SCENARIOS.items()
               if name != "all_conditions")
