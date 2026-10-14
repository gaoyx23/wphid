from experiments import paper_h2_optional_sharing_controls as experiment


def test_h2_contains_true_sharing_positive_and_false_sharing_negative_controls():
    cases = experiment.PROTOCOL["truth_cases"]
    assert cases["shared_positive"]["truth_partition"] == "shared"
    assert cases["independent_negative"]["truth_partition"] == "independent"
    assert cases["shared_positive"]["temperature"] == cases["shared_positive"]["oxygen"]
    assert cases["independent_negative"]["temperature"] != cases["independent_negative"]["oxygen"]
    assert experiment.SHARED_GROUPS == (0, 0, None)
    assert experiment.INDEPENDENT_GROUPS == (0, 1, None)


def test_h2_confirmation_seeds_are_disjoint_from_development():
    assert set(experiment.PROTOCOL["development_seeds"]).isdisjoint(
        experiment.PROTOCOL["confirmation_seeds"]
    )


def test_h2_condition_folds_are_disjoint_and_cover_all_conditions():
    folds = [set(values) for values in experiment.PROTOCOL["condition_folds"]]
    assert folds[0].isdisjoint(folds[1])
    assert set().union(*folds) == set(experiment.CONDITIONS)
