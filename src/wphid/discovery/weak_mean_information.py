"""Conditional local mean sensitivity with full shared-noise covariance.

Covariance derivatives are not included: this is not full Fisher information,
calibrated uncertainty or global identifiability.
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import vstack


def whitened_mean_jacobian(states, operators, bases, parameters, coefficients, groups, noise_stds):
    states, parameters, coefficients = np.asarray(states).reshape(3, -1), np.asarray(parameters), np.asarray(coefficients)
    temperature, oxygen, damage = states
    count = operators["mass"].shape[0]
    parameter_count = parameters.size
    dimension = parameter_count+12
    rates, gradients, jacobian = [], [], np.zeros((3*count, dimension))
    mass = operators["mass"]
    scales = np.maximum(np.abs(coefficients), 1e-12)
    for equation, group in enumerate(groups):
        theta, m, q = parameters[group]
        rate = np.exp(-theta/temperature)*oxygen**m*(1-damage)**q
        rates.append(rate)
        gradients.append((rate*theta/temperature**2, rate*m/oxygen, -rate*q/(1-damage)))
        section = slice(equation*count, (equation+1)*count)
        shapes = (-rate/temperature, rate*np.log(oxygen), rate*np.log1p(-damage))
        for index, derivative in enumerate(shapes):
            jacobian[section, 3*group+index] = -coefficients[equation, 3]*(mass@derivative)*parameters[group, index]
        for index in range(3):
            jacobian[section, parameter_count+4*equation+index] = -bases[equation][:, index]*scales[equation, index]
        jacobian[section, parameter_count+4*equation+3] = -(mass@rate)*scales[equation, 3]
    covariance = np.zeros((3*count, 3*count))
    for field, noise in enumerate(noise_stds):
        blocks = []
        for equation in range(3):
            block = -coefficients[equation, 3]*mass.multiply(gradients[equation][field][None, :])
            if equation == field:
                block = (block-operators["time"]-coefficients[equation, 0]*operators["axial"]
                         -coefficients[equation, 1]*operators["radial"]-coefficients[equation, 2]*mass)
            blocks.append(block)
        noise_jacobian = vstack(blocks, format="csr")*noise
        covariance += (noise_jacobian@noise_jacobian.T).toarray()
    row_scale = 1/np.sqrt(np.diag(covariance))
    covariance *= row_scale[:, None]*row_scale[None, :]
    covariance += 1e-10*np.mean(np.diag(covariance))*np.eye(len(covariance))
    factor = cholesky(covariance, lower=True)
    return solve_triangular(factor, jacobian*row_scale[:, None], lower=True)


def mean_information_summary(jacobian, amplitude_coordinate):
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    if singular[-1] <= singular[0]*1e-12:
        return {"full_numerical_rank": False, "singular_values": singular.tolist(), "amplitude_relative_scale_proxy": None}
    inverse = (right.T/singular**2)@right
    return {"full_numerical_rank": True, "singular_values": singular.tolist(),
            "amplitude_relative_scale_proxy": float(np.sqrt(inverse[amplitude_coordinate, amplitude_coordinate])),
            "information_matrix": (jacobian.T@jacobian).tolist(),
            "qualification": "Local mean-only sensitivity proxy in relative coordinates, not a calibrated interval or full parameter-dependent-covariance Fisher bound."}
