import numpy as np
import pytest
from wphid.discovery.centered_reaction import observed_pivot, encode_strength, decode_amplitude, centered_action


@pytest.mark.parametrize("amplitude", [4500., -11., 7.])
def test_exact_coordinate_identity_and_roundtrip(amplitude):
    states = np.array([[800., 1100., 1500.], [.2, .6, .8], [.1, .3, .5]])
    parameters = [7000., .65, 1.25]
    pivot = observed_pivot(states)
    sign, strength = encode_strength(amplitude, parameters, pivot)
    assert decode_amplitude(sign, strength, parameters, pivot) == pytest.approx(amplitude)
    expected = amplitude*np.exp(-7000/states[0])*states[1]**.65*(1-states[2])**1.25
    np.testing.assert_allclose(centered_action(states, sign, strength, parameters, pivot), expected, rtol=1e-13)


def test_changing_pivot_changes_coordinates_not_physics():
    states = np.array([[800., 1100.], [.2, .6], [.1, .3]])
    parameters = [7000., .65, 1.25]
    first = observed_pivot(states)
    second = first+np.array([1e-4, .1, .2])
    one = encode_strength(-11., parameters, first)
    two = encode_strength(-11., parameters, second)
    np.testing.assert_allclose(centered_action(states, *one, parameters, first), centered_action(states, *two, parameters, second))


def test_empty_mechanism_not_hidden_in_log_active_coordinates():
    with pytest.raises(ValueError):
        encode_strength(0., [7000, .65, 1.25], np.zeros(3))
