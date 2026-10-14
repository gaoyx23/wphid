from experiments.paper_r99_passive_cooldown_acoustic_observability import STATE_NAMES


def test_r99_cooldown_inherits_every_dynamic_state_required_by_truth_solver():
    assert STATE_NAMES == ("T", "pO2", "pH2O", "pProducts", "carbon", "crack")
