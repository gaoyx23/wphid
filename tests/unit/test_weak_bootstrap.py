import numpy as np
import pytest
import torch
from test_joint_weak_likelihood import problem
from wphid.discovery.weak_bootstrap import centered_measurement_bootstrap


def test_zero_noise_centers_residual_without_mutating_original_observations():
    model = problem()
    original = model.packets[0]["targets"].clone()
    replica = centered_measurement_bootstrap(model, model.initial, 55, noise_multiplier=0.)
    assert replica.decomposition(model.initial)["quadratic_sum"] < 1e-20
    torch.testing.assert_close(model.packets[0]["targets"], original)
    assert replica.packets[0]["targets"] is not model.packets[0]["targets"]


def test_same_nodal_noise_reaches_both_dictionary_and_target():
    model = problem()
    centered = centered_measurement_bootstrap(model, model.initial, 55, noise_multiplier=0.)
    replica = centered_measurement_bootstrap(model, model.initial, 55)
    p, base = replica.packets[0], model.packets[0]
    delta = p["states"]-base["states"]
    torch.testing.assert_close(p["targets"]-centered.packets[0]["targets"], -delta@base["time"].T)
    expected = torch.stack([delta@base[k].T for k in ("axial", "radial", "mass")], dim=-1)
    torch.testing.assert_close(p["bases"]-base["bases"], expected)
    np.testing.assert_array_equal(replica.packets[0]["noise"].numpy(), base["noise"].numpy())


def test_reproducible_noise_and_no_silent_redraw():
    model = problem()
    one = centered_measurement_bootstrap(model, model.initial, 55)
    two = centered_measurement_bootstrap(model, model.initial, 55)
    torch.testing.assert_close(one.packets[0]["states"], two.packets[0]["states"])
    with pytest.raises(ValueError):
        centered_measurement_bootstrap(model, model.initial, 55, noise_multiplier=-1.)
