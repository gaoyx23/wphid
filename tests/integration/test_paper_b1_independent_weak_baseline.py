import experiments.paper_b1_independent_weak_baseline as experiment


def test_independent_weak_baseline_returns_one_family_per_equation():
    row = experiment.run_seed(31801, True)
    assert len(row["selected_families"]) == len(experiment.FIELDS)
    assert set(row["selected_families"]).issubset(set(experiment.FAMILIES))
