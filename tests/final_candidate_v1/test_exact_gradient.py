"""Independent differentiation checks; no scientific observations or scores."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from wphid.final_candidate_v1.problem import WeakCondition, TrainingScale, paper_families
from wphid.final_candidate_v1.exact_gradient import ExactGradientView


@pytest.mark.parametrize("family", paper_families(), ids=lambda f: f.name)
def test_exact_cpu_weak_jacobian_matches_full_autograd_at_two_parameter_points(family):
    states = (np.array([700., 950., 1200., 1500.]), np.array([.2, 1.3, .7, 1.1]),
              np.array([.08, .4, .6, .1]))
    projection = csr_matrix(np.array([[.1, .2, .3, .4], [.4, .3, .2, .1]]))
    condition = WeakCondition("literal", (np.ones((2,1)),)*3,
                              (np.array([1.,3.]), np.array([3.,1.]), np.array([2.,4.])), states, projection)
    scale = TrainingScale.fit((condition,), (family,))
    slow = scale.apply((condition,)).torch_builder(family)
    fast = ExactGradientView((condition,), scale).torch_builder(family)
    for fraction in (.31, .72):
        values = [lo+fraction*(hi-lo) for lo, hi in family.bounds]
        p = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        a, b = slow(p), fast(p)
        for left, right in zip(a,b):
            np.testing.assert_allclose(left.detach().numpy(), right.detach().numpy(), rtol=3e-14, atol=1e-14)
        weight = torch.tensor([.7, -.4], dtype=torch.float64)
        ga = torch.autograd.grad(a[1]@weight, p)[0]
        gb = torch.autograd.grad(b[1]@weight, p)[0]
        np.testing.assert_allclose(ga.numpy(), gb.numpy(), rtol=2e-13, atol=1e-14)


def test_exact_path_does_not_depend_on_heldout_states():
    family = paper_families()[-3]
    condition = WeakCondition("train", (np.ones((2,1)),)*3, (np.array([1.,2.]),)*3,
                              (np.array([900.,1300.]), np.array([.4,1.1]), np.array([.1,.6])), csr_matrix(np.eye(2)))
    scale = TrainingScale.fit((condition,), (family,))
    original = scale.fingerprint
    fast = ExactGradientView((condition,), scale).torch_builder(family)
    p = torch.tensor(family.center, dtype=torch.float64, requires_grad=True)
    fast(p)[0].sum().backward()
    assert torch.isfinite(p.grad).all() and scale.fingerprint == original
