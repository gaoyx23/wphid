import numpy as np
import pytest
from wphid.types import Candidate, DiscoveryResult, Trajectory


def test_trajectory_rejects_mismatched_time_axis():
    with pytest.raises(ValueError, match="time axis"):
        Trajectory(np.arange(3.0), {"Do": np.zeros(2)}, "c0")


def test_trajectory_rejects_non_1d_time_axis():
    with pytest.raises(ValueError, match="one-dimensional"):
        Trajectory(np.zeros((2, 2)), {}, "c0")


def test_trajectory_rejects_non_increasing_time_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(np.array([0.0, 1.0, 1.0]), {}, "c0")


def test_trajectory_rejects_scalar_state():
    with pytest.raises(ValueError, match="time axis"):
        Trajectory(np.arange(3.0), {"Do": np.array(1.0)}, "c0")


def test_trajectory_copies_and_freezes_mutable_inputs():
    source_t = np.arange(3.0)
    source_state = np.array([1.0, 2.0, 3.0])
    source_states = {"Do": source_state}
    trajectory = Trajectory(source_t, source_states, "c0")

    source_t[0] = 99.0
    source_state[0] = 99.0
    source_states["new"] = np.zeros(3)

    np.testing.assert_array_equal(trajectory.t, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(trajectory.states["Do"], [1.0, 2.0, 3.0])
    assert "new" not in trajectory.states

    with pytest.raises(ValueError, match="read-only"):
        trajectory.t[0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        trajectory.states["Do"][0] = 5.0
    with pytest.raises(TypeError):
        trajectory.states["new"] = np.zeros(3)


def test_candidate_copies_mutable_metadata():
    dimension = [1, 0]
    parents = ["T"]
    coefficient_dimension = [-1, 0]
    candidate = Candidate(
        "T",
        lambda states: states["T"],
        dimension,
        "F",
        parents,
        coefficient_dimension=coefficient_dimension,
        canonical_identity="state:T",
    )

    dimension[0] = 9
    parents.append("pO2")
    coefficient_dimension[0] = 9

    assert candidate.dimension == (1, 0)
    assert candidate.parents == ("T",)
    assert candidate.coefficient_dimension == (-1, 0)
    assert candidate.canonical_identity == "state:T"


def test_discovery_result_copies_and_freezes_mutable_inputs():
    source_coefficients = np.array([1.0, 2.0])
    source_support = np.array([0])
    source_diagnostics = {"condition": "c0"}
    result = DiscoveryResult(
        ("T", "pO2"),
        source_coefficients,
        source_support,
        0.1,
        source_diagnostics,
    )

    source_coefficients[0] = 99.0
    source_support[0] = 1
    source_diagnostics["condition"] = "c1"
    source_diagnostics["new"] = True

    np.testing.assert_array_equal(result.coefficients, [1.0, 2.0])
    np.testing.assert_array_equal(result.support, [0])


@pytest.mark.parametrize(
    ("names", "coefficients", "support", "validation_error", "message"),
    [
        (("x", "x"), np.ones(2), np.array([0]), 0.1, "unique"),
        (("x",), np.ones(2), np.array([0]), 0.1, "one coefficient"),
        (("x",), np.array([[1.0]]), np.array([0]), 0.1, "one-dimensional"),
        (("x",), np.array([np.nan]), np.array([], dtype=int), 0.1, "finite"),
        (("x",), np.ones(1), np.array([True]), 0.1, "integer indices"),
        (("x",), np.ones(1), np.array([1]), 0.1, "within"),
        (("x", "y"), np.ones(2), np.array([1, 0]), 0.1, "strictly increasing"),
        (("x",), np.ones(1), np.array([0]), -0.1, "non-negative"),
        (("x",), np.ones(1), np.array([0]), np.inf, "finite"),
    ],
)
def test_discovery_result_rejects_malformed_contracts(
    names, coefficients, support, validation_error, message
):
    """Malformed public results otherwise corrupt support and gate metrics."""
    with pytest.raises(ValueError, match=message):
        DiscoveryResult(
            names,
            coefficients,
            support,
            validation_error,
            {},
        )
