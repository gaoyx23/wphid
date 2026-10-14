from experiments.p2_e3_frustum_generalization import GEOMETRIES, PROTOCOL


def test_p2_e3_protocol_declares_cross_geometry_matrix():
    assert PROTOCOL["version"] == "P2_E3_FRUSTUM_GENERALIZATION_V3_EXCESS_RESIDUAL"
    assert set(GEOMETRIES) == {"nominal", "shallow", "steep", "thin_tapered", "thick_tapered"}
    assert PROTOCOL["grid_shapes"] == [[33, 33, 129], [41, 41, 161]]
    assert not set(PROTOCOL["seeds"]) & set(PROTOCOL["historical_diagnostic_seeds"])
