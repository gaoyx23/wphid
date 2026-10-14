from experiments import paper_m4_hierarchical_stable_two_mechanism as experiment


def test_hierarchical_stability_uses_fresh_seed_ranges():
    assert set(experiment.PROTOCOL["development_seeds"]).isdisjoint(
        experiment.PROTOCOL["confirmation_seeds"]
    )
    assert min(experiment.PROTOCOL["development_seeds"]) > 36605
