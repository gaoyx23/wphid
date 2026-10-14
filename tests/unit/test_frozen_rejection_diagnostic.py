import pytest
from experiments.audit_frozen_paired_rejections import attribute_true_rejection, maximum_calibration_reference


@pytest.mark.parametrize("pair,absolute,cause",[(1,1,"accepted"),(3,1,"paired_only"),(1,3,"absolute_only"),(3,3,"both")])
def test_component_attribution_preserves_joint_decision(pair,absolute,cause):
    value=attribute_true_rejection(dict(squared_radius=2,contrast_scores={"other":pair},
        absolute_score=absolute,inside=max(pair,absolute)<=2))
    assert value["cause"]==cause


def test_inconsistent_saved_decision_rejected():
    with pytest.raises(ValueError):
        attribute_true_rejection(dict(squared_radius=2,contrast_scores={"b":3},absolute_score=1,inside=True))


def test_reference_distribution_normalization_and_mean():
    tails=[maximum_calibration_reference(32,128,k)["probability_at_least_observed_rejections"] for k in range(129)]
    assert tails[0]==pytest.approx(1)
    assert all(a>=b for a,b in zip(tails,tails[1:]))
    assert sum(tails[1:])==pytest.approx(128/33)
    assert maximum_calibration_reference(32,128,9)["probability_conditional_coverage_below_95_percent"]==pytest.approx(.95**32)


def test_invalid_reference_counts_rejected():
    with pytest.raises(ValueError):
        maximum_calibration_reference(0,128,9)
