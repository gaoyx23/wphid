import numpy as np
import torch
from test_joint_weak_likelihood import problem
from wphid.discovery.latent_balance_correction import correction_summary
from wphid.discovery.weak_bootstrap import centered_measurement_bootstrap


def test_minimum_noise_energy_matches_quadratic_and_measurements_unchanged():
    model = problem()
    original = [p["states"].clone() for p in model.packets]
    # Small weak inconsistency avoids using a deliberately impossible random fixture.
    centered = centered_measurement_bootstrap(model, model.initial, 4, noise_multiplier=.1)
    result = correction_summary(centered, model.initial)
    assert result["all_function_domains_preserved"]
    np.testing.assert_allclose(result["correction_energy"], result["original_quadratic"], rtol=1e-7)
    for row in result["packets"]:
        assert row["linear_balance_remaining_relative"] < 1e-7
        assert row["nonlinear_remaining_whitened_norm_relative"] < .001
    for p, old in zip(model.packets, original):
        torch.testing.assert_close(p["states"], old)


def test_zero_balance_requires_zero_correction():
    model = problem()
    zero = centered_measurement_bootstrap(model, model.initial, 4, noise_multiplier=0.)
    result = correction_summary(zero, model.initial)
    assert result["correction_energy"] < 1e-20
