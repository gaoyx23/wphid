import numpy as np

from experiments.paper_o4_axial_boundary_sensitivity import (
    PROTOCOL,
    _boundary_influx,
    _laplacian,
)


def test_non_insulated_boundaries_apply_nonzero_end_exchange():
    values = np.full((9, 9), 1100.0)
    left, right = _boundary_influx("robin_exchange", 0.22, 0, values)
    assert np.any(left != 0.0)
    assert np.any(right != 0.0)
    assert PROTOCOL["minimum_endpoint_temperature_excursion"] > 0


def test_insulated_operator_annihilates_constant_field():
    values = np.full((9, 9), 3.0)
    zero = np.zeros(9)
    axial, radial = _laplacian(
        values, 0.125, 0.125, np.linspace(0.0, 1.0, 9), zero, zero
    )
    assert np.allclose(axial, 0.0)
    assert np.allclose(radial, 0.0)
