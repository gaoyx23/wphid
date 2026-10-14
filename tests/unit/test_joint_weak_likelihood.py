import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from wphid.discovery.joint_weak_likelihood import ArrheniusJointWeakLikelihood


def problem():
    torch.set_num_threads(1)
    rng = np.random.default_rng(122)
    n, windows = 18, 4
    states = np.vstack((rng.uniform(900, 1200, n), rng.uniform(0.3, 0.9, n), rng.uniform(0.1, 0.5, n)))
    operators = {name: csr_matrix(rng.normal(size=(windows, n)))
                 for name in ("mass", "time", "axial", "radial")}
    bases = np.asarray([np.column_stack([operators[k]@u for k in ("axial", "radial", "mass")]) for u in states])
    targets = np.asarray([-operators["time"]@u for u in states])
    params = np.array([[7000, 0.65, 1.25]])
    coefficients = np.array([[0.1, 0.05, -0.18, 4500], [0.07, 0.04, -0.12, -11], [0.03, 0.02, -0.08, 7]])
    packet = {"states": states, "operators": operators, "bases": bases, "targets": targets,
              "noise_stds": [0.1, 1e-4, 1e-4]}
    return ArrheniusJointWeakLikelihood([packet], (0, 0, 0),
        ((3500, 14000), (0.2, 2), (0.3, 3)), params, coefficients)


def test_autodiff_agrees_with_finite_difference_for_shapes_and_coefficients():
    model = problem()
    value, gradient = model.value_and_gradient(model.initial)
    assert np.isfinite(value)
    for coordinate in (0, 1, 2, 3, 6, 10, 14):
        left, right = model.initial.copy(), model.initial.copy()
        left[coordinate] -= 1e-5
        right[coordinate] += 1e-5
        difference = (model.value_and_gradient(right)[0]-model.value_and_gradient(left)[0])/2e-5
        assert gradient[coordinate] == pytest.approx(difference, rel=2e-5, abs=0.01)


def test_decomposition_and_physical_coordinates_are_consistent():
    model = problem()
    value, _ = model.value_and_gradient(model.initial)
    components = model.decomposition(model.initial)
    assert value == pytest.approx(components["normalized_half_nll"])
    params, coefficients = model.physical_values(model.initial)
    np.testing.assert_allclose(params, [[7000, 0.65, 1.25]])
    assert coefficients.shape == (3, 4)
    assert components["cross_equation_covariance_used"]


def test_joint_likelihood_matches_independent_covariance_construction():
    model = problem()
    packet = model.packets[0]
    parameters, coefficients = model.physical_values(model.initial)
    states = packet["states"].numpy()
    mass, time, axial, radial = [packet[k].numpy() for k in ("mass", "time", "axial", "radial")]
    theta, m, q = parameters[0]
    temperature, oxygen, damage = states
    rate = np.exp(-theta/temperature)*oxygen**m*(1-damage)**q
    gradients = [rate*theta/temperature**2, rate*m/oxygen, -rate*q/(1-damage)]
    covariance = np.zeros((12, 12))
    for field in range(3):
        rows = []
        for equation in range(3):
            row = -coefficients[equation, 3]*mass*gradients[field][None, :]
            if equation == field:
                row = row-time-coefficients[equation, 0]*axial-coefficients[equation, 1]*radial-coefficients[equation, 2]*mass
            rows.append(row)
        jacobian = np.vstack(rows)*packet["noise"].numpy()[field]
        covariance += jacobian@jacobian.T
    assert np.max(np.abs(covariance[:4, 4:])) > 0
    scale = packet["row_scale"].numpy().ravel()
    covariance = covariance*scale[:, None]*scale[None, :]
    covariance += model.relative_jitter*np.trace(covariance)/len(covariance)*np.eye(12)
    residual = (packet["targets"].numpy()-np.einsum("ewk,ek->ew", packet["bases"].numpy(), coefficients[:, :3])
                -coefficients[:, 3, None]*(mass@rate)[None, :]).ravel()*scale
    factor = np.linalg.cholesky(covariance)
    expected = (np.log(np.diag(factor)).sum()+0.5*np.sum(np.linalg.solve(factor, residual)**2))/12
    assert model.value_and_gradient(model.initial)[0] == pytest.approx(expected, rel=1e-10)


def test_parameter_changes_update_covariance_logdet():
    model = problem()
    initial = model.decomposition(model.initial)
    changed = model.initial.copy()
    changed[6] *= 1.3  # Temperature amplitude, not a fixed noise matrix.
    updated = model.decomposition(changed)
    assert initial["logdet_sum"] != pytest.approx(updated["logdet_sum"], abs=1e-8)


@pytest.mark.parametrize("groups", [(0, 0, 1), (0, 1, 2)])
def test_distinct_parameter_groups_match_independent_joint_construction(groups):
    original = problem()
    packet = original.packets[0]
    raw = {"states": packet["states"].numpy(), "noise_stds": packet["noise"].numpy(),
           "operators": {k: csr_matrix(packet[k].numpy()) for k in ("mass", "time", "axial", "radial")},
           "bases": packet["bases"].numpy(), "targets": packet["targets"].numpy()}
    params = np.array([[7000, .65, 1.25], [9000, 1.1, .8], [6000, .5, 1.8]])[:max(groups)+1]
    coefficients = original.physical_values(original.initial)[1]
    model = ArrheniusJointWeakLikelihood([raw], groups,
        ((3500, 14000), (.2, 2), (.3, 3)), params, coefficients)
    temperature, oxygen, damage = raw["states"]
    theta, m, q = params[np.asarray(groups)].T
    rate = np.exp(-theta[:, None]/temperature)*oxygen**m[:, None]*(1-damage)**q[:, None]
    gradients = np.stack((rate*theta[:, None]/temperature**2,
                          rate*m[:, None]/oxygen, -rate*q[:, None]/(1-damage)), axis=1)
    mass, time, axial, radial = [raw["operators"][k].toarray() for k in ("mass", "time", "axial", "radial")]
    covariance = np.zeros((12, 12))
    for field in range(3):
        rows = []
        for equation in range(3):
            row = -coefficients[equation, 3]*mass*gradients[equation, field][None, :]
            if equation == field:
                row = row-time-coefficients[equation, 0]*axial-coefficients[equation, 1]*radial-coefficients[equation, 2]*mass
            rows.append(row)
        jacobian = np.vstack(rows)*raw["noise_stds"][field]
        covariance += jacobian@jacobian.T
    scale = model.packets[0]["row_scale"].numpy().ravel()
    covariance *= scale[:, None]*scale[None, :]
    covariance += model.relative_jitter*np.trace(covariance)/12*np.eye(12)
    residual = (raw["targets"]-np.einsum("ewk,ek->ew", raw["bases"], coefficients[:, :3])
                -coefficients[:, 3, None]*(rate@mass.T)).ravel()*scale
    factor = np.linalg.cholesky(covariance)
    expected = (np.log(np.diag(factor)).sum()+.5*np.sum(np.linalg.solve(factor, residual)**2))/12
    assert model.value_and_gradient(model.initial)[0] == pytest.approx(expected, rel=1e-10)
