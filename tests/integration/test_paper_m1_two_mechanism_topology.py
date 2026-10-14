from experiments import paper_m1_two_mechanism_topology as experiment


def test_two_mechanism_manufactured_network_has_matched_weak_systems():
    local, bases, targets = experiment._assemble(36101, 1.0, 1.0)
    assert len(local) == 4
    assert len(bases) == len(targets) == 3
    assert all(base.shape[0] == target.size for base, target in zip(bases, targets))
    assert all(base.shape[1] == 3 for base in bases)


def test_refinement_budget_preserves_each_model_stratum():
    rows = []
    for reaction in experiment.TOPOLOGIES:
        for leakage in experiment.TOPOLOGIES:
            rows.append({"reaction_topology": list(reaction),
                         "leakage_topology": list(leakage),
                         "screen_bic": float(len(rows))})
    selected = experiment._select_finalists(rows)
    counts = {
        name: sum(experiment._refinement_stratum(row) == name for row in selected)
        for name in experiment.PROTOCOL["refinement_quota"]
    }
    assert counts == experiment.PROTOCOL["refinement_quota"]
