import numpy as np

from experiments.s3b_e1_closed_coupled_network import _clean


def test_closed_network_has_three_finite_fields_and_forcings():
    _, _, _, states, forcing = _clean("mixed")
    assert len(states) == len(forcing) == 3
    assert all(np.all(np.isfinite(field)) for field in (*states, *forcing))
    assert np.min(states[0]) > 500.0
    assert np.min(states[1]) > 0.0
    assert 0.0 <= np.min(states[2]) <= np.max(states[2]) < 1.0
