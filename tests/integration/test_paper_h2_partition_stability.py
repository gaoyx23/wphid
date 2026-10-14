from experiments import paper_h2_partition_stability as experiment


def test_v3_retains_v2_no_go_and_uses_new_disjoint_seeds():
    assert experiment.PROTOCOL["supersedes_for_development"].endswith("V2")
    old = experiment.v2.PROTOCOL
    new = experiment.PROTOCOL
    old_seeds = set(old["development_seeds"] + old["confirmation_seeds"])
    new_seeds = set(new["development_seeds"] + new["confirmation_seeds"])
    assert old_seeds.isdisjoint(new_seeds)


def test_v3_refits_four_leave_one_excitation_out_folds():
    assert len(experiment.v2.CONDITIONS) == 4
    assert experiment.PROTOCOL["minimum_condition_support"] == 0.75
    assert "leave_one_excitation_out" in experiment.PROTOCOL["folds"]
    assert experiment.PROTOCOL["tie_support_floor"] == 0.50
    assert experiment.PROTOCOL["minimum_signed_fold_bic_sum"] > 0.0


def test_v3_contains_shared_and_independent_controls():
    cases = experiment.PROTOCOL["truth_cases"]
    assert cases["shared_positive"]["truth_partition"] == "shared"
    assert cases["independent_negative"]["truth_partition"] == "independent"
