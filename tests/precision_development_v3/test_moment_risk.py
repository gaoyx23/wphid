"""Independent identities and failure conditions on artificial fixtures."""

import numpy as np
import pytest

from wphid.final_candidate_v1.observations import adapt_condition, reaction_projection
from wphid.final_candidate_v1.problem import TrainingScale, paper_families
from wphid.precision_development_v3.moment_risk import (
    WeakMomentRisk, arrhenius_derivatives, solve_corrected_quadratic, weak_operators,
)


def fixture():
    t, z, r = np.linspace(0, 2, 9), np.linspace(0, 1, 9), np.linspace(0, .7, 9)
    a, b, c = np.meshgrid(t, z, r, indexing="ij")
    states = (850 + 140*a + 19*np.cos(2.4*b) + 8*np.cos(3*c) + 3*a*b*b,
              .16 + .025*np.sin(2*a) + .011*np.cos(2*b) + .004*c*c + .003*a*c*c,
              .12 + .035*a + .012*np.cos(3*b) + .007*np.cos(2.2*c) + .004*a*b)
    forcing = tuple(np.sin(a)*np.cos(b) + (i+1)*c*c + .3*i for i in range(3))
    condition = adapt_condition(name="artificial", t=t, z=z, r=r,
                                observed_states=states, supplied_forcing=forcing,
                                window_shape=(5, 5, 5), stride_shape=(2, 2, 2))
    family = next(f for f in paper_families() if f.name == "arrhenius_oxygen_damage")
    scale = TrainingScale.fit((condition,), (family,))
    ops = weak_operators(t, z, r, (5, 5, 5), (2, 2, 2))
    return condition, family, scale.apply((condition,)), ops


@pytest.mark.parametrize("equation", [0, 1, 2])
def test_operators_match_independent_saved_adapter(equation):
    condition, _, _, ops = fixture()
    state = condition.states[equation]
    np.testing.assert_allclose(np.column_stack([op @ state for op in ops[:3]]),
                               condition.bases[equation], rtol=1e-12, atol=1e-11)


def test_reaction_projection_identity():
    t, z, r = np.linspace(0, 2, 9), np.linspace(0, 1, 9), np.linspace(0, .7, 9)
    ops = weak_operators(t, z, r, (5, 5, 5), (2, 2, 2))
    np.testing.assert_allclose(ops[2].toarray(), reaction_projection(t, z, r, (5, 5, 5), (2, 2, 2)).toarray(),
                               rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("field", [0, 1, 2])
def test_analytic_state_derivatives_against_finite_differences(field):
    states = (np.array([1100., 1250.]), np.array([.17, .23]), np.array([.12, .27]))
    p = np.array([7000., .65, 1.25])
    rate, first, second = arrhenius_derivatives(states, p)
    h = np.array([.03, 1e-5, 1e-5])[field]
    plus, minus = [x.copy() for x in states], [x.copy() for x in states]
    plus[field] += h
    minus[field] -= h
    rp = arrhenius_derivatives(plus, p)[0]
    rm = arrhenius_derivatives(minus, p)[0]
    np.testing.assert_allclose((rp-rm)/(2*h), first[field], rtol=2e-7, atol=1e-13)
    np.testing.assert_allclose((rp-2*rate+rm)/h**2, second[field], rtol=1e-4, atol=1e-10)


def test_contracted_noise_gram_equals_full_independent_jacobians():
    condition, family, view, ops = fixture()
    sigma = np.array([.2, 2e-5, 3e-5])
    risk = WeakMomentRisk(view, family, (0, 0, 0), ops, (sigma,))
    p = np.array([7100., .7, 1.2])
    _, gradients, _ = arrhenius_derivatives(condition.states, p)
    for e in range(3):
        _, gamma, cross, yy = risk.equation(e, p)
        w = view.scale.target_weights[e]
        bs = w / view.scale.base_norms[e]
        rs = w / view.scale.norms_for(family)[e]
        column_jacobians = []
        for j in range(4):
            blocks = []
            for field in range(3):
                if j == 3:
                    block = ops[2].multiply(gradients[field]) * rs
                else:
                    block = ops[j] * (bs[j] if field == e else 0.)
                blocks.append(block.toarray() * sigma[field])
            column_jacobians.append(np.column_stack(blocks))
        jy = np.column_stack([ops[3].toarray() * (w*sigma[field] if field == e else 0.)
                              for field in range(3)])
        full_gram = np.array([[np.sum(a*b) for b in column_jacobians] for a in column_jacobians])
        full_cross = np.array([np.sum(a*jy) for a in column_jacobians])
        np.testing.assert_allclose(gamma, full_gram, rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(cross, full_cross, rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(yy, np.sum(jy*jy), rtol=2e-13)


def test_zero_noise_is_exact_ordinary_quadratic_profile():
    _, family, view, ops = fixture()
    risk = WeakMomentRisk(view, family, (0, 1, 2), ops, (np.zeros(3),))
    p = np.tile([7100., .7, 1.2], 3)
    for e in range(3):
        a, gamma, cross, yy = risk.equation(e, p, True)
        b = risk.equation(e, p, False)[0]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(gamma, np.zeros((4, 4)))
        np.testing.assert_array_equal(cross, np.zeros(4))
        assert yy == 0.
    # The tiny weak fixture can be ill-conditioned; use an independently
    # well-resolved matrix for the linear solver's zero-noise identity.
    x = np.linspace(-1, 1, 15)
    matrix = np.column_stack((np.ones(15), x, x*x, np.sin(2*x)))
    target = .3 + .7*x + .2*np.cos(3*x)
    c, value, _, _ = solve_corrected_quadratic(matrix, target, np.zeros((4, 4)), np.zeros(4))
    ordinary = np.linalg.lstsq(matrix, target, rcond=None)[0]
    np.testing.assert_allclose(c, ordinary, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(value, np.sum((target-matrix@ordinary)**2), rtol=1e-12, atol=1e-12)


def test_exact_linear_noise_moments_remove_expected_score_bias():
    # Symmetric deterministic quadrature, not random scientific realizations.
    x0 = np.array([[1., -.5], [1., .5], [1., 1.5], [1., 2.5]])
    truth = np.array([.4, 1.3])
    y0 = x0 @ truth
    dx = np.zeros_like(x0)
    dx[:, 1] = np.array([.2, -.1, .15, -.05])
    dy = np.array([.1, -.2, .3, .15])
    exx = .5 * ((x0+dx).T@(x0+dx) + (x0-dx).T@(x0-dx))
    exy = .5 * ((x0+dx).T@(y0+dy) + (x0-dx).T@(y0-dy))
    corrected = np.linalg.solve(exx-dx.T@dx, exy-dx.T@dy)
    np.testing.assert_allclose(corrected, truth, rtol=1e-13, atol=1e-13)
    assert np.linalg.norm(np.linalg.solve(exx, exy)-truth) > 1e-3


@pytest.mark.parametrize("factor", [1., 2.])
def test_indefinite_or_singular_gram_is_rejected_without_clipping(factor):
    x = np.eye(3)
    with pytest.raises(ValueError, match="positive definite"):
        solve_corrected_quadratic(x, np.ones(3), factor*np.eye(3), np.zeros(3))


def test_negative_corrected_risk_is_retained():
    x = np.array([[1.], [1.]])
    c, value, _, _ = solve_corrected_quadratic(x, np.ones(2), np.zeros((1, 1)), np.zeros(1), gamma_yy=.1)
    np.testing.assert_allclose(c, [1.])
    assert value == pytest.approx(-.1)


def test_invalid_noise_covariance_is_rejected():
    _, family, view, ops = fixture()
    with pytest.raises(ValueError, match="nonnegative"):
        WeakMomentRisk(view, family, (0, 0, 0), ops, ([-1., 0., 0.],))
