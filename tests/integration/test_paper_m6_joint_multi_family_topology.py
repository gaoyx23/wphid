from experiments import paper_m6_joint_multi_family_topology as experiment


def test_joint_screen_preserves_all_model_strata():
    rows = []
    for reaction_family, reaction_topology in experiment._options(
            experiment.m5.REACTION_FAMILIES):
        for leakage_family, leakage_topology in experiment._options(
                experiment.m5.LEAKAGE_FAMILIES):
            rows.append({"reaction_family": reaction_family,
                         "reaction_topology": list(reaction_topology),
                         "leakage_family": leakage_family,
                         "leakage_topology": list(leakage_topology),
                         "screen_bic": float(len(rows))})
    selected = experiment._select_finalists(rows)
    counts = {name: sum(experiment._stratum(row) == name for row in selected)
              for name in experiment.PROTOCOL["refinement_quota"]}
    assert counts == experiment.PROTOCOL["refinement_quota"]
