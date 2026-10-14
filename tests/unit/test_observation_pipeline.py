from dataclasses import replace

import numpy as np
import pytest

from wphid.discovery.observation_pipeline import (
    ForwardTemplate, ObservationPacket, identify_observation_packet,
    profile_cached_templates,
)
from wphid.discovery.unified import WeakProposal


def packet():
    return ObservationPacket(np.array([2., 4.]), np.diag([1., 4.]),
                             ("port:oxygen:t=0", "port:steam:t=0"), ("Pa", "Pa"),
                             "measured-port-v1", "hotfire-001")


def template(candidate, nuisance, values):
    p = packet()
    return ForwardTemplate(candidate, nuisance, np.array(values), p.labels,
                           p.units, p.operator_id, p.condition_id)


def test_gls_score_and_nuisance_profile():
    scores, selected = profile_cached_templates(packet(), ("a", "b"), [
        template("a", "k-low", [1., 2.]), template("a", "k-high", [2., 4.]),
        template("b", "k-low", [1., 2.]),
    ])
    assert scores == {"a": 0., "b": 1.}
    assert selected["a"] == "k-high"


def test_correlated_covariance_not_diagonal_approximation():
    p = replace(packet(), covariance=np.array([[1., .5], [.5, 1.]]))
    scores, _ = profile_cached_templates(p, ("a",), [template("a", "n", [1., 3.])])
    assert scores["a"] == pytest.approx(2./3.)


def test_content_fingerprint_includes_values_covariance_units_and_condition():
    p = packet()
    for changed in [replace(p, values=np.array([3., 4.])),
                    replace(p, covariance=np.eye(2)), replace(p, units=("kPa", "Pa")),
                    replace(p, condition_id="other")]:
        assert p.observation_id != changed.observation_id
    assert p.observation_id == packet().observation_id


def test_arrays_are_copied_and_read_only():
    values = np.array([2., 4.]); p = replace(packet(), values=values)
    values[0] = 999
    assert p.values[0] == 2
    with pytest.raises(ValueError):
        p.values[0] = 0


@pytest.mark.parametrize("changes", [
    {"labels": ("steam", "oxygen")}, {"units": ("kPa", "Pa")},
    {"operator_id": "other"}, {"condition_id": "other"},
    {"prediction": np.array([2.])},
])
def test_physically_incompatible_templates_rejected(changes):
    with pytest.raises(ValueError, match="incompatible"):
        profile_cached_templates(packet(), ("a",), [replace(template("a", "n", [2., 4.]), **changes)])


def test_same_packet_callback_and_forward_scoring():
    p = packet()
    calls = []
    def propose(observations):
        calls.append(observations)
        return WeakProposal(observations.observation_id, ("b",), True, "unit-test-only")
    result = identify_observation_packet(p, candidate_ids=("a", "b"),
        templates=[template("a", "n", [2., 4.]), template("b", "n", [1., 2.])],
        gap_threshold=.1, calibration_id="unit-test-only", adequacy_upper_bound=2.,
        weak_adapter=propose)
    assert calls == [p]
    assert result["candidate_set"] == ["a", "b"]
    assert result["status"] == "stage_conflict"


def test_missing_candidate_template_withholds_result():
    result = identify_observation_packet(packet(), candidate_ids=("a", "b"),
        templates=[template("a", "n", [2., 4.])], gap_threshold=.1,
        calibration_id="unit-test-only", adequacy_upper_bound=2.)
    assert result["status"] == "incomplete_forward"


@pytest.mark.parametrize("changes", [
    {"covariance": np.zeros((2, 2))}, {"covariance": np.array([[1., 2.], [0., 1.]])},
    {"values": np.array([float("nan"), 2.])}, {"labels": ("a", "a")},
    {"units": ("", "Pa")},
])
def test_invalid_observation_packets_rejected(changes):
    with pytest.raises(ValueError):
        replace(packet(), **changes)
