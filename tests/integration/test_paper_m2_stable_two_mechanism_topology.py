from experiments import paper_m2_stable_two_mechanism_topology as experiment


def test_edge_stability_uses_condition_level_evidence_and_signs():
    raw = experiment.m1.run_seed(36101, 1.0, 1.0)
    diagnostics, qualified = experiment._edge_stability(36101, 1.0, 1.0, raw)
    assert len(diagnostics) == 5
    assert all(len(row["condition_bic_gains"]) == 4 for row in diagnostics)
    assert set(qualified) == {"reaction", "leakage"}
