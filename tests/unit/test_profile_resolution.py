import pytest
from wphid.discovery.profile_resolution import resolution_witness


def test_first_incompatible_budget_is_derived_not_an_arbitrary_cutoff():
    points = [{"amplitude": .75, "score_cost": .4}, {"amplitude": 1., "score_cost": 0.},
              {"amplitude": 1.25, "score_cost": .24}, {"amplitude": 1.5, "score_cost": .8}]
    result = resolution_witness(points, .25)
    assert result["minimum_sampled_score_budget_for_incompatible_point_targets"] == .8
    witness = result["witness"]
    assert witness["first"]["amplitude"] == .75 and witness["second"]["amplitude"] == 1.5


def test_negative_amplitudes_use_signed_intervals_and_same_result():
    result = resolution_witness([{"amplitude": -.75, "score_cost": .4}, {"amplitude": -1.5, "score_cost": .8}], .25)
    assert result["witness"]["nonoverlap_gap"] > 0
    assert result["minimum_sampled_score_budget_for_incompatible_point_targets"] == .8


def test_touching_intervals_are_not_declared_impossible():
    result = resolution_witness([{"amplitude": .75, "score_cost": .4}, {"amplitude": 1.25, "score_cost": .2}], .25)
    assert result["witness"] is None


def test_opposite_signs_are_incompatible_but_zero_relative_error_is_undefined():
    assert resolution_witness([{"amplitude": -1., "score_cost": .1}, {"amplitude": 1., "score_cost": .2}], .25)["witness"]
    with pytest.raises(ValueError):
        resolution_witness([{"amplitude": 0., "score_cost": .1}, {"amplitude": 1., "score_cost": .2}], .25)
