import json
from pathlib import Path
import numpy as np
import pytest
from experiments.audit_joint_reaction_prefactor_profile import OUTPUT,NAMES


@pytest.fixture
def saved():
    if not OUTPUT.exists():pytest.skip("saved profile unavailable")
    result=json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["complete"]
    return result


def test_clean_joint_prefactors_and_transport_are_consistent(saved):
    for source in NAMES:
        row=saved["clean"][source]
        assert row["minimum_score_transport"]==source
        fit=row["profiles"][source]["oxygen_and_steam"]
        np.testing.assert_allclose(fit["amplitude_multipliers"],[1.,1.],atol=.003)
        assert fit["normalized_design_rank"]==2
        assert not fit["calibrated_support_selection"]


def test_all_saved_packets_profiled_without_new_noise_or_claimed_intervals(saved):
    assert saved["new_noise_draws"]==saved["new_forward_runs"]==0
    assert saved["completed_draws"]==dict.fromkeys(NAMES,128)
    for source in NAMES:
        assert saved["noisy"][source]["observation_count"]==128
        assert sum(saved["noisy"][source]["minimum_score_counts"].values())==128
        assert not saved["noisy"][source]["parameter_coverage_established"]
    for path in (OUTPUT.parent/"draws").rglob("*.json"):
        row=json.loads(path.read_text(encoding="utf-8"))
        assert row["parameter_output"] is None
        for fits in row["profiles"].values():
            full=fits["oxygen_and_steam"]["weighted_squared_residual"]
            for fit in fits.values():
                assert full<=fit["weighted_squared_residual"]+1.e-8
                assert fit["parameter_confidence_intervals"] is None


def test_bias_diagnostic_reproduces_counts_without_changing_paired_rule():
    path=OUTPUT.parents[1]/"reaction_profile_score_bias/audit.json"
    if not path.exists():pytest.skip("saved bias diagnostic unavailable")
    report=json.loads(path.read_text(encoding="utf-8"))
    assert report["complete"]
    assert not report["original_paired_decisions_changed"]
    assert report["new_parameter_fits"]==0
    for source,row in report["results"].items():
        assert sum(row["fixed_prefactors_minimum_score_counts"].values())==128
        assert sum(row["fitted_prefactors_minimum_score_counts"].values())==128
