"""Tests for pre-inference Arrhenius excitation diagnostics."""

import numpy as np
import pytest

from wphid.truth.aging_ode import simulate_named_condition
from wphid.truth.identifiability import diagnose_arrhenius_identifiability


def test_mission_profiles_report_temperature_span_and_arrhenius_diagnostics():
    trajectories = tuple(
        simulate_named_condition(condition_id)
        for condition_id in ("thermal_hold", "fatigue_rich", "oxygen_rich")
    )

    diagnostics = diagnose_arrhenius_identifiability(
        trajectories, theta_candidates=(1_200.0, 1_800.0, 2_400.0)
    )

    assert diagnostics["temperature_span"]["minimum"] >= 300.0
    assert diagnostics["temperature_span"]["maximum"] <= 1_500.0
    assert diagnostics["temperature_span"]["span"] >= 700.0
    assert diagnostics["standardized_condition_number"] > 1.0
    assert np.isfinite(diagnostics["standardized_condition_number"])
    assert diagnostics["arrhenius_column_correlation"].shape == (3, 3)
    assert np.allclose(
        np.diag(diagnostics["arrhenius_column_correlation"]), 1.0
    )
    assert diagnostics["arrhenius_column_correlation"][0, 2] < 0.9999
    assert diagnostics["profile_score_separation"] > 0.0


def test_arrhenius_diagnostics_reject_invalid_candidates_and_missing_temperature():
    trajectory = simulate_named_condition("thermal_hold")

    with pytest.raises(ValueError, match="unique"):
        diagnose_arrhenius_identifiability((trajectory,), theta_candidates=(1800, 1800))

    with pytest.raises(ValueError, match="positive"):
        diagnose_arrhenius_identifiability((trajectory,), theta_candidates=(0, 1800))

    with pytest.raises(ValueError, match="at least two"):
        diagnose_arrhenius_identifiability((), theta_candidates=(1200, 1800))
