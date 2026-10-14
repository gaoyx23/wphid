import numpy as np
import pytest
from wphid.discovery.profile_information import diagnose_profile_information, conditional_profile_objective


def test_amplitude_like_parameter_is_projected_out():
    x = np.linspace(0.2, 2, 60)
    bases = (np.ones((60, 1)),)
    def builder(p):
        return (p[0]*x**p[1],)
    targets = (1+3*x**1.4 + 0.001*np.sin(13*x),)
    result = diagnose_profile_information(bases, targets, builder, ([2.0, 1.4],), (0,))
    assert result[0]["smallest_to_largest_singular_ratio"] < 1e-8
    assert abs(result[0]["weakest_relative_parameter_direction"][0]) > 0.99


def test_profile_objective_is_amplitude_invariant():
    x = np.linspace(0.2, 2, 60)
    bases, targets = (np.ones((60, 1)),), (1+3*x**1.4 + 0.001*np.sin(13*x),)
    left = conditional_profile_objective(bases, targets, (x**1.4,))
    right = conditional_profile_objective(bases, targets, (700*x**1.4,))
    assert left == pytest.approx(right, abs=1e-8)


def test_independent_groups_report_separately():
    x = np.linspace(0.2, 2, 60)
    bases = (np.ones((60, 1)),)*2
    def builder(p):
        return (x**p[0], x**p[0])
    targets = (1+3*x**1.4, 2+7*x**0.7)
    result = diagnose_profile_information(bases, targets, builder, ([1.4], [0.7]), (0, 1))
    assert [r["equations"] for r in result] == [[0], [1]]
    with pytest.raises(ValueError):
        diagnose_profile_information(bases, targets, builder, ([0.0], [0.7]), (0, 1))
