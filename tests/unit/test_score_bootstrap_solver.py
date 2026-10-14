import numpy as np
from experiments.run_h3_score_bootstrap_pilot import fit_bootstrap
from wphid.discovery.curvature_coordinates import whitening_coordinates


class QuadraticModel:
    kinetic_count = 2
    center = np.array([.5, .4, 2.3, 1.])
    matrix = np.array([[3., .1, .2, 0.], [.1, 2., .1, 0.], [.2, .1, 1., .3], [0., 0., .3, 2.]])

    def value_and_gradient(self, vector):
        difference = vector-self.center
        return .5*difference@self.matrix@difference, self.matrix@difference


def test_free_and_restricted_pair_matches_analytic_conditional_quadratic():
    model = QuadraticModel()
    rotation = whitening_coordinates(model.matrix)
    reference = np.array([.5, .4, 2., 1.])
    free = fit_bootstrap(model, reference, rotation)
    start = np.asarray(free["vector"])
    start[2] = reference[2]
    restricted = fit_bootstrap(model, start, rotation, fixed_coordinate=2)
    assert free["optimizer_success"] and restricted["optimizer_success"]
    assert free["nll"] <= restricted["nll"]
    assert abs(restricted["vector"][2]-2.) < 1e-8
    indices = np.array([0, 1, 3])
    expected = model.center.copy()
    expected[2] = 2.
    expected[indices] -= np.linalg.solve(model.matrix[np.ix_(indices, indices)], model.matrix[indices, 2]*(-.3))
    np.testing.assert_allclose(restricted["vector"], expected, atol=1e-7)
    assert restricted["tangent_transformed_gradient_norm"] < 1e-7
