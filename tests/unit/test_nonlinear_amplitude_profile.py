"""Synthetic constrained quadratic checks; not scientific profile samples."""
import numpy as np
from experiments import run_h3_nonlinear_amplitude_profile as runner
from wphid.discovery.curvature_coordinates import whitening_coordinates


class QuadraticModel:
    kinetic_count = 2
    center = np.array([.5, .4, 2., 1.])
    matrix = np.array([[3., .1, .2, 0.], [.1, 2., .1, 0.], [.2, .1, 1., .3], [0., 0., .3, 2.]])

    def value_and_gradient(self, point):
        difference = point-self.center
        return .5*difference@self.matrix@difference, self.matrix@difference

    def physical_values(self, point):
        return point[:2].reshape(1, 2), point[2:].reshape(1, 2)


def test_fixed_amplitude_profiles_nuisance_not_just_a_straight_line():
    model = QuadraticModel()
    rotation = whitening_coordinates(model.matrix)
    result = runner.point_fit(model, model.center, rotation, 2, 1.25)
    free = np.array([0, 1, 3])
    expected = model.center.copy()
    expected[2] = 2.5
    expected[free] -= np.linalg.solve(model.matrix[np.ix_(free, free)], model.matrix[free, 2]*.5)
    reconstructed = np.r_[result["parameters"][0], result["coefficients"][0]]
    np.testing.assert_allclose(reconstructed, expected, atol=1e-7)
    assert result["optimizer_success"]
    assert result["nll"] < model.value_and_gradient(np.array([.5, .4, 2.5, 1.]))[0]
    assert result["equality_tangent_coordinate_gradient_norm"] < 1e-7


def test_negative_amplitude_ratio_keeps_its_sign():
    model = QuadraticModel()
    model.center = np.array([.5, .4, -2., 1.])
    result = runner.point_fit(model, model.center, whitening_coordinates(model.matrix), 2, .75)
    assert abs(result["coefficients"][0][0]+1.5) < 1e-8


def test_budget_failure_retained_without_restart(monkeypatch):
    monkeypatch.setitem(runner.PROTOCOL, "max_evaluations", 1)
    model = QuadraticModel()
    result = runner.point_fit(model, model.center, whitening_coordinates(model.matrix), 2, 1.5)
    assert not result["optimizer_success"] and result["objective_evaluations"] == 1
    assert "budget" in result["optimizer_message"]
