import pytest

from wphid.discovery.unified import WeakProposal, reconcile_identification


def identify(**changes):
    args = dict(observation_id="measured-001", candidate_ids=("a", "b", "c"),
                forward_scores={"a": .1, "b": .3, "c": 1.2},
                gap_threshold=.25, calibration_id="development-v1",
                adequacy_upper_bound=1.0)
    args.update(changes)
    return reconcile_identification(**args)


def test_forward_set_never_narrowed_by_weak_proposal():
    p = WeakProposal("measured-001", ("a",), True, "weak-test")
    r = identify(proposal=p)
    assert r["candidate_set"] == ["a", "b"]
    assert r["status"] == "stage_conflict"
    assert not r["scientific_confirmation"]


def test_conflict_preserves_both_explanations():
    p = WeakProposal("measured-001", ("c",), True, "weak-test")
    assert identify(proposal=p)["candidate_set"] == ["a", "b", "c"]


def test_different_observation_packets_rejected():
    with pytest.raises(ValueError, match="different observations"):
        identify(proposal=WeakProposal("other-packet", ("a",), True, "weak-test"))


def test_missing_score_prevents_unique_claim():
    r = identify(forward_scores={"a": .1})
    assert r["status"] == "incomplete_forward"
    assert r["candidate_set"] == ["a", "b", "c"]


def test_failed_fit_is_not_silently_removed():
    assert identify(forward_scores={"a": .1, "b": float("nan"), "c": 1.2})["status"] == "incomplete_forward"


def test_absolute_failure_withholds_library_interpretation():
    r = identify(adequacy_upper_bound=.05)
    assert r["status"] == "model_inadequate"
    assert r["candidate_set"] == []
    assert r["library_review_required"]


def test_missing_adequacy_calibration_is_not_success():
    assert identify(adequacy_upper_bound=None)["status"] == "adequacy_uncalibrated"


def test_parameter_requires_unique_structure_and_identifiability():
    estimates = {"a": {"coefficient": 2.}}
    assert identify(parameter_identifiable=True, parameter_estimates=estimates)["parameter_output"] is None
    assert identify(gap_threshold=.01, parameter_estimates=estimates)["parameter_output"] is None
    assert identify(gap_threshold=.01, parameter_estimates=estimates,
                    parameter_identifiable=True)["parameter_output"] == estimates["a"]


def test_conflict_withholds_numeric_parameters_even_for_one_candidate():
    p = WeakProposal("measured-001", (), True, "weak-test")
    r = identify(gap_threshold=.01, proposal=p, parameter_identifiable=True,
                 parameter_estimates={"a": {"coefficient": 2.}})
    assert r["status"] == "stage_conflict"
    assert r["parameter_output"] is None


def test_incomplete_proposal_only_routes_does_not_select():
    p = WeakProposal("measured-001", ("c",), False, "weak-test")
    assert identify(proposal=p)["candidate_set"] == ["a", "b"]


def test_zero_threshold_is_explicitly_flagged():
    assert identify(gap_threshold=0)["zero_gap_hard_ranking"]


@pytest.mark.parametrize("changes", [
    {"gap_threshold": -1}, {"gap_threshold": float("nan")},
    {"adequacy_upper_bound": -1}, {"candidate_ids": ("a", "a")},
    {"calibration_id": ""}, {"forward_scores": {"undeclared": .1}},
    {"proposal": WeakProposal("measured-001", ("d",), True, "weak-test")},
])
def test_invalid_contracts_rejected(changes):
    with pytest.raises(ValueError):
        identify(**changes)
