import numpy as np
import pytest
from wphid.final_candidate_v1.problem import WeakCondition, TrainingScale, paper_families
from wphid.precision_development_v3.moment_risk import weak_operators
from wphid.wendy_comparison_v1.gaussian_weak import GaussianWeakProblem


def fixture(groups=(0,0,0)):
    grid = (np.linspace(0., 2., 5), np.linspace(.1, 1., 5), np.linspace(.2, 1., 5))
    ops = weak_operators(*grid, window_shape=(3,3,3), stride_shape=(2,2,2))
    rng = np.random.default_rng(7301)
    states = (1000.+100*rng.random(125), .8+.2*rng.random(125), .1+.2*rng.random(125))
    condition = WeakCondition('software', tuple(np.column_stack([o@s for o in ops[:3]]) for s in states),
                              tuple(ops[3]@s for s in states), states, ops[2])
    family = next(f for f in paper_families() if f.name=='arrhenius_oxygen_damage')
    scale = TrainingScale.fit((condition,), (family,))
    problem = GaussianWeakProblem(scale.apply((condition,)), family, groups, ops, ((.5,.002,.001),))
    vector = np.r_[np.full(problem.theta_count, .4), np.tile([.4,-.2,.1,.25],3)]
    return problem, vector, ops, condition


@pytest.mark.parametrize('groups', [(0,0,0), (0,0,1), (0,1,2)])
def test_full_joint_gradient_including_covariance_and_logdet(groups):
    problem, vector, _, _ = fixture(groups)
    value, analytic = problem.value_gradient(vector)
    difference = []
    for k in range(len(vector)):
        step = 2e-6
        plus, minus = vector.copy(), vector.copy()
        plus[k] += step
        minus[k] -= step
        difference.append((problem.value_gradient(plus)[0]-problem.value_gradient(minus)[0])/(2*step))
    np.testing.assert_allclose(analytic, difference, rtol=3e-5, atol=1e-2)
    assert np.isfinite(value)


def test_covariance_retains_cross_equation_correlations_and_raw_jacobian():
    problem, vector, ops, condition = fixture()
    value, gradient, blocks = problem.value_gradient(vector, return_blocks=True)
    factor = blocks[0]['noise_factor'].toarray()
    covariance = blocks[0]['covariance']
    np.testing.assert_allclose(covariance, factor@factor.T, rtol=1e-12, atol=1e-15)
    assert np.linalg.norm(covariance[:8,8:16])>0
    # Independently perturb one raw oxygen node in the residual expression.
    e, node, epsilon = 0, 31, 1e-6
    theta = problem.decode(vector[:3])
    physical = problem.physical_coefficients(vector)
    def residual(states):
        rates = problem.family.values(states, theta)
        return np.concatenate([(sum(c[k]*(ops[k]@states[e]) for k in range(3))
                                +c[3]*(ops[2]@rates)-ops[3]@states[e])*problem.training.scale.target_weights[e]
                               for e,c in enumerate(physical)])
    plus, minus = [s.copy() for s in condition.states], [s.copy() for s in condition.states]
    plus[1][node] += epsilon
    minus[1][node] -= epsilon
    numeric = (residual(plus)-residual(minus))/(2*epsilon)*problem.sigma[0][1]
    np.testing.assert_allclose(factor[:,125+node], numeric, rtol=2e-6, atol=1e-10)


def test_scaled_coefficients_preserve_raw_physical_residual():
    problem, vector, ops, condition = fixture()
    _, _, blocks = problem.value_gradient(vector, return_blocks=True)
    physical = problem.physical_coefficients(vector)
    rate = problem.family.values(condition.states, problem.decode(vector[:3]))
    raw = np.concatenate([(sum(c[k]*(ops[k]@condition.states[e]) for k in range(3))
                           +c[3]*(ops[2]@rate)-condition.targets[e])*problem.training.scale.target_weights[e]
                          for e,c in enumerate(physical)])
    np.testing.assert_allclose(raw, blocks[0]['residual'], rtol=2e-13, atol=1e-12)


def test_invalid_partition_and_outside_box_rejected():
    problem, vector, ops, condition = fixture()
    with pytest.raises(ValueError, match='unit box'):
        problem.decode([-1e-10,.4,.4])
    with pytest.raises(ValueError, match='consecutive'):
        GaussianWeakProblem(problem.training, problem.family, (0,0,2), ops, ((.5,.002,.001),))


def test_rank_failure_retained_without_regularization():
    problem, vector, _, _ = fixture()
    # Zero all parameters and c gives only storage; duplicate rows make it singular.
    problem.operators = tuple(type(op)(np.tile(op.toarray()[:1], (8,1))) for op in problem.operators)
    with pytest.raises(np.linalg.LinAlgError):
        problem.value_gradient(vector)
