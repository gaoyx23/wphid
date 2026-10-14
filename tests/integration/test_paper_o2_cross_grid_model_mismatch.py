from experiments.paper_o2_cross_grid_model_mismatch import PROTOCOL, _truth


def test_truth_grid_differs_from_discovery_grid_and_contains_mismatch():
    assert PROTOCOL["truth_grid_shape"] != PROTOCOL["discovery_grid_shape"]
    assert PROTOCOL["axial_diffusivity_gradient"] > 0
    t, z, r, states, forcing = _truth("mixed", 1.0, 1.0)
    expected = tuple(PROTOCOL["truth_grid_shape"])
    assert (len(t), len(z), len(r)) == expected
    assert all(field.shape == expected for field in (*states, *forcing))
