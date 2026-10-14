import numpy as np
import pytest

from wphid.discovery.conditioning import diagnose_mechanism_information


def test_duplicate_atom_has_no_independent_information():
    x = np.linspace(-1, 1, 25)
    row = diagnose_mechanism_information(x[:, None], 3 * x)
    assert row["independent_energy_fraction"] < 1e-25
    assert row["conditional_amplitude_noise_gain"] is None


def test_information_fraction_is_invariant_to_units():
    x = np.linspace(-1, 1, 25)
    base = np.column_stack((np.ones_like(x), x))
    atom = x**2
    left = diagnose_mechanism_information(base, atom)
    right = diagnose_mechanism_information(base * [1e9, 1e-9], atom * 7)
    assert left["independent_energy_fraction"] == pytest.approx(right["independent_energy_fraction"])
    assert right["conditional_amplitude_noise_gain"] * 7 == pytest.approx(left["conditional_amplitude_noise_gain"])


def test_condition_wise_confounding_does_not_imply_pooled_confounding():
    # Each condition alone: mechanism is indistinguishable from base. Across
    # conditions the slopes differ and a common coefficient is identifiable.
    x = np.arange(1.0, 6.0)
    assert diagnose_mechanism_information(x[:, None], 2 * x)["conditional_amplitude_noise_gain"] is None
    assert diagnose_mechanism_information(x[:, None], 5 * x)["conditional_amplitude_noise_gain"] is None
    pooled = diagnose_mechanism_information(np.concatenate((x, x))[:, None],
                                             np.concatenate((2*x, 5*x)))
    assert pooled["independent_energy_fraction"] > 0.1


def test_zero_atom_and_invalid_input():
    assert diagnose_mechanism_information(np.zeros((4, 0)), np.zeros(4))["atom_is_zero"]
    with pytest.raises(ValueError):
        diagnose_mechanism_information(np.zeros((3, 1)), np.zeros(4))
