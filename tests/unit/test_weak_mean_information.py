import numpy as np
from scipy.sparse import csr_matrix, vstack
from test_joint_weak_likelihood import problem
from wphid.discovery.weak_mean_information import whitened_mean_jacobian


def inputs():
    model = problem()
    packet = model.packets[0]
    parameters, coefficients = model.physical_values(model.initial)
    return [packet["states"].numpy(), {k: csr_matrix(packet[k].numpy()) for k in ("mass", "time", "axial", "radial")},
            packet["bases"].numpy(), parameters, coefficients, model.groups, packet["noise"].numpy()]


def test_noise_doubling_halves_whitened_mean_sensitivity():
    original = inputs()
    doubled = list(original)
    doubled[-1] = original[-1]*2
    np.testing.assert_allclose(whitened_mean_jacobian(*doubled), whitened_mean_jacobian(*original)/2, rtol=1e-10)


def test_duplicate_tests_do_not_double_information_from_same_observations():
    original = inputs()
    duplicated = list(original)
    duplicated[1] = {k: vstack([matrix, matrix], format="csr") for k, matrix in original[1].items()}
    duplicated[2] = np.concatenate((original[2], original[2]), axis=1)
    one, two = whitened_mean_jacobian(*original), whitened_mean_jacobian(*duplicated)
    np.testing.assert_allclose(two.T@two, one.T@one, rtol=1e-7, atol=1e-5)
