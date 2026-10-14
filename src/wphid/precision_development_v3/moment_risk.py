"""Second-order corrected quadratic risk at fixed nonlinear parameters.

The independent Gaussian state noise covariance must be supplied. The weak
right-hand side and library columns inherit correlated errors from the same
states. Merely correcting the mean nonlinear column omits their Gram/cross
moment corrections. This module implements both, without fitting a latent
field, resampling data, optimizing nonlinear parameters, or clipping a bad
corrected Gram matrix to make it positive.

This is a local Taylor approximation, not an exact Gaussian deconvolution or
a likelihood, confidence certificate, or parameter-accuracy guarantee.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import csr_matrix

from wphid.weak.axisymmetric import axisymmetric_window_slices
from wphid.weak.axisymmetric_library import _bump_with_derivatives


def weak_operators(t, z, r, window_shape=(15, 15, 15), stride_shape=(5, 5, 5)):
    """Linear operators [axial, radial, self, target] with V1 quadrature."""
    coordinates = tuple(np.asarray(x, dtype=np.float64) for x in (t, z, r))
    if any(x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all()
           or not np.all(np.diff(x) > 0) for x in coordinates) or coordinates[2][0] < 0:
        raise ValueError("Invalid axisymmetric grid")
    shape = tuple(len(x) for x in coordinates)
    indices_grid = np.arange(np.prod(shape)).reshape(shape)
    indices, values, offsets = [], [[], [], [], []], [0]
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        local = tuple(x[s] for x, s in zip(coordinates, slices))
        wt, wz, wr = (simpson(np.eye(len(x)), x=x, axis=1) for x in local)
        quadrature = np.einsum("i,j,k->ijk", wt, wz, wr)
        bt, dt, _ = _bump_with_derivatives(local[0])
        bz, _, ddz = _bump_with_derivatives(local[1])
        br, dr, ddr = _bump_with_derivatives(local[2])
        radial = local[2][None, None, :]
        tests = (
            bt[:, None, None] * ddz[None, :, None] * br[None, None, :] * radial,
            bt[:, None, None] * bz[None, :, None]
            * (radial * ddr[None, None, :] + dr[None, None, :]),
            bt[:, None, None] * bz[None, :, None] * br[None, None, :] * radial,
            -dt[:, None, None] * bz[None, :, None] * br[None, None, :] * radial,
        )
        indices.extend(indices_grid[slices].ravel())
        for buffer, test in zip(values, tests):
            buffer.extend((quadrature * test).ravel())
        offsets.append(len(indices))
    if len(offsets) == 1:
        raise ValueError("No weak windows")
    return tuple(csr_matrix((np.asarray(value), np.asarray(indices), np.asarray(offsets)),
                            shape=(len(offsets) - 1, int(np.prod(shape)))) for value in values)


def arrhenius_derivatives(states, parameters):
    """Rate, nodewise first derivatives and diagonal second derivatives."""
    t, o, d = (np.asarray(x, dtype=np.float64) for x in states)
    p = np.asarray(parameters, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all() or any(
            x.ndim != 1 or x.shape != t.shape or not np.isfinite(x).all() for x in (t, o, d)):
        raise ValueError("Invalid Arrhenius input")
    if np.any(t <= 0) or np.any(o <= 0) or np.any(d >= 1):
        raise ValueError("Arrhenius derivatives outside domain")
    theta, m, q = p
    rate = np.exp(-theta / t) * o**m * (1 - d)**q
    gradients = (rate * theta / t**2, rate * m / o, -rate * q / (1 - d))
    diagonal = (rate * (theta**2 / t**4 - 2 * theta / t**3),
                rate * m * (m - 1) / o**2,
                rate * q * (q - 1) / (1 - d)**2)
    return rate, gradients, diagonal


@dataclass(frozen=True)
class ProfilePoint:
    objective: float
    coefficients: tuple[np.ndarray, ...]
    gram_margins: tuple[float, ...]
    relative_gram_corrections: tuple[float, ...]
    matrices: tuple[np.ndarray, ...]


def solve_corrected_quadratic(matrix, target, gamma_xx, gamma_xy, gamma_yy=0.,
                              minimum_relative_eigenvalue=1e-10):
    """Finite linear profile; reject indefinite/poorly resolved Gram matrices."""
    matrix, target = np.asarray(matrix), np.asarray(target)
    gram = matrix.T @ matrix
    gamma_xx, gamma_xy = np.asarray(gamma_xx), np.asarray(gamma_xy)
    if (matrix.ndim != 2 or target.shape != (len(matrix),)
            or gamma_xx.shape != gram.shape or gamma_xy.shape != (gram.shape[0],)
            or not all(np.isfinite(x).all() for x in (matrix, target, gamma_xx, gamma_xy))
            or not np.isfinite(gamma_yy) or gamma_yy < 0
            or not np.allclose(gamma_xx, gamma_xx.T, rtol=1e-12, atol=1e-15)):
        raise ValueError("Invalid corrected quadratic inputs")
    corrected = gram - gamma_xx
    largest = float(np.linalg.eigvalsh(gram)[-1])
    smallest = float(np.linalg.eigvalsh(corrected)[0])
    margin = smallest / max(largest, np.finfo(float).tiny)
    if margin <= minimum_relative_eigenvalue:
        raise ValueError("Corrected Gram matrix is not sufficiently positive definite")
    c = np.linalg.solve(corrected, matrix.T @ target - gamma_xy)
    residual = target - matrix @ c
    noise_energy = gamma_yy - 2 * c @ gamma_xy + c @ gamma_xx @ c
    # An unbiased quadratic-risk estimate can be negative; no log or floor.
    value = float(residual @ residual - noise_energy)
    relative = float(np.linalg.norm(gamma_xx, 2) / max(largest, np.finfo(float).tiny))
    return c, value, margin, relative


class WeakMomentRisk:
    """Fixed single-reaction structure, scales, windows and state covariance."""

    def __init__(self, training, family, groups, operators, sigma_by_condition):
        if family.name != "arrhenius_oxygen_damage" or len(groups) != 3 or any(g is None for g in groups):
            raise ValueError("This preflight covers only three active Arrhenius destinations")
        if set(groups) != set(range(max(groups) + 1)) or len(sigma_by_condition) != len(training.conditions):
            raise ValueError("Invalid fixed partition or noise layout")
        self.view, self.family, self.groups = training, family, tuple(groups)
        self.operators = tuple(operators)
        self.sigma = tuple(np.asarray(s, dtype=float) for s in sigma_by_condition)
        if any(s.shape != (3,) or not np.isfinite(s).all() or np.any(s < 0) for s in self.sigma):
            raise ValueError("Noise standard deviations must be finite and nonnegative")
        projection = self.operators[2]
        for condition in training.conditions:
            if projection.shape != condition.projection.shape or not np.allclose(
                    projection.toarray(), condition.projection.toarray(), rtol=2e-13, atol=1e-15):
                raise ValueError("Reaction projection differs from the saved input")
            for e, state in enumerate(condition.states):
                projected = np.column_stack([op @ state for op in self.operators[:3]])
                if not np.allclose(projected, condition.bases[e], rtol=2e-11, atol=2e-12):
                    raise ValueError("Base operators differ from saved weak columns")
        # Diagonal contractions suffice for independent node/field noise.
        self.pp = np.asarray(projection.multiply(projection).sum(axis=0)).ravel()
        self.bp = tuple(np.asarray(op.multiply(projection).sum(axis=0)).ravel()
                        for op in self.operators[:3])
        self.py = np.asarray(projection.multiply(self.operators[3]).sum(axis=0)).ravel()
        self.bb = np.asarray([[float(a.multiply(b).sum()) for b in self.operators[:3]]
                              for a in self.operators[:3]])
        self.by = np.asarray([float(op.multiply(self.operators[3]).sum()) for op in self.operators[:3]])
        self.yy = float(self.operators[3].multiply(self.operators[3]).sum())

    def equation(self, e, parameters, mean_correction=True):
        parameters = np.asarray(parameters).reshape(max(self.groups) + 1, 3)
        p = parameters[self.groups[e]]
        atom, gamma = [], np.zeros((4, 4))
        cross, yy = np.zeros(4), 0.
        w = self.view.scale.target_weights[e]
        bn = self.view.scale.base_norms[e]
        rn = self.view.scale.norms_for(self.family)[e]
        bs, rs = w / bn, w / rn
        for condition, sigma in zip(self.view.conditions, self.sigma):
            rate, gradients, diagonal = arrhenius_derivatives(condition.states, p)
            bias = .5 * sum(s*s*h for s, h in zip(sigma, diagonal))
            atom.append(np.asarray(condition.projection @ (rate - bias if mean_correction else rate)) * rs)
            gamma[:3, :3] += sigma[e]**2 * self.bb * np.outer(bs, bs)
            for j in range(3):
                value = sigma[e]**2 * bs[j] * rs * (self.bp[j] @ gradients[e])
                gamma[j, 3] += value
                gamma[3, j] += value
            gamma[3, 3] += rs**2 * sum(s*s*(self.pp @ (g*g)) for s, g in zip(sigma, gradients))
            cross[:3] += sigma[e]**2 * bs * w * self.by
            cross[3] += sigma[e]**2 * rs * w * (self.py @ gradients[e])
            yy += sigma[e]**2 * w*w * self.yy
        matrix = np.column_stack((self.view.bases[e], np.concatenate(atom)))
        return matrix, gamma, cross, yy

    def profile(self, parameters, mean_correction=True, gram_correction=True):
        coefficients, matrices, margins, relative = [], [], [], []
        objective = 0.
        for e, target in enumerate(self.view.targets):
            matrix, gamma, cross, yy = self.equation(e, parameters, mean_correction)
            if gram_correction:
                c, value, margin, correction = solve_corrected_quadratic(matrix, target, gamma, cross, yy)
            else:
                c = np.linalg.lstsq(matrix, target, rcond=None)[0]
                residual = target - matrix @ c
                value = float(residual @ residual)
                eigen = np.linalg.eigvalsh(matrix.T @ matrix)
                margin = float(eigen[0] / eigen[-1])
                correction = 0.
            objective += value
            coefficients.append(c)
            matrices.append(matrix)
            margins.append(margin)
            relative.append(correction)
        return ProfilePoint(objective, tuple(coefficients), tuple(margins), tuple(relative), tuple(matrices))


def saved_point_preflight(risk, parameters, bounds, step=1e-5):
    """Fixed-point finite differences and a local Gauss--Newton response only.

    Predicted steps are never applied, constrained, selected, or graded as fits.
    The new Gram correction is separated from the previously studied mean term.
    """
    parameters, bounds = np.asarray(parameters), np.asarray(bounds)
    lower, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    unit = (parameters - lower) / span
    if np.any(unit - step <= 0) or np.any(unit + step >= 1):
        raise ValueError("Saved point does not admit the declared interior tangent check")
    raw = risk.profile(parameters, False, False)
    mean = risk.profile(parameters, True, False)
    corrected = risk.profile(parameters, True, True)
    width = len(parameters)
    jacobian, coefficient_jacobian = [], np.zeros((12, width))
    score_difference = np.zeros(width)
    for k in range(width):
        delta = np.zeros(width)
        delta[k] = step * span[k]
        mp, mm = risk.profile(parameters + delta, True, False), risk.profile(parameters - delta, True, False)
        cp, cm = risk.profile(parameters + delta, True, True), risk.profile(parameters - delta, True, True)
        score_difference[k] = ((cp.objective - mp.objective) - (cm.objective - mm.objective)) / (2 * step)
        coefficient_jacobian[:, k] = (np.concatenate(mp.coefficients) - np.concatenate(mm.coefficients)) / (2 * step)
        tangent = []
        for e, (plus, minus, matrix, c) in enumerate(zip(mp.matrices, mm.matrices, mean.matrices, mean.coefficients)):
            t = (plus - minus) @ c / (2 * step)
            tangent.append(t - matrix @ np.linalg.lstsq(matrix, t, rcond=None)[0])
        jacobian.append(np.concatenate(tangent))
    tangent = np.column_stack(jacobian)
    singular = np.linalg.svd(tangent, compute_uv=False)
    tolerance = np.finfo(float).eps * max(tangent.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != width:
        raise ValueError("Projected tangent is rank deficient at saved point")
    hessian = 2 * tangent.T @ tangent
    predicted_step = -np.linalg.solve(hessian, score_difference)
    direct = np.concatenate(corrected.coefficients) - np.concatenate(mean.coefficients)
    predicted_coefficient = direct + coefficient_jacobian @ predicted_step
    mean_coefficients = np.concatenate(mean.coefficients)
    amplitudes = np.array([3, 7, 11])
    if np.any(np.abs(mean_coefficients[amplitudes]) <= 1e-12):
        raise ValueError("Zero mean-only amplitude has no relative-shift diagnostic")
    amplitude_direct = direct[amplitudes] / np.abs(mean_coefficients[amplitudes])
    amplitude_predicted = predicted_coefficient[amplitudes] / np.abs(mean_coefficients[amplitudes])
    max_step = float(np.max(np.abs(predicted_step)))
    return {
        "status": "computed",
        "raw_quadratic_profile": raw.objective,
        "mean_only_quadratic_profile": mean.objective,
        "corrected_quadratic_profile": corrected.objective,
        "minimum_corrected_gram_relative_eigenvalue": min(corrected.gram_margins),
        "relative_gram_corrections": list(corrected.relative_gram_corrections),
        "projected_tangent_rank": rank,
        "projected_tangent_condition_number": float(singular[0] / singular[-1]),
        "correction_score_gradient_in_unit_box": score_difference.tolist(),
        "predicted_internal_parameter_shift_in_unit_box": predicted_step.tolist(),
        "maximum_predicted_unit_box_shift": max_step,
        "direct_conditional_amplitude_fractional_shifts": amplitude_direct.tolist(),
        "predicted_amplitude_fractional_shifts": amplitude_predicted.tolist(),
        "maximum_predicted_amplitude_fractional_shift": float(np.max(np.abs(amplitude_predicted))),
        "mean_only_conditional_scaled_coefficients": [c.tolist() for c in mean.coefficients],
        "corrected_conditional_scaled_coefficients": [c.tolist() for c in corrected.coefficients],
        "local_response_within_declared_radius": max_step <= .1,
        "finite_difference_step_unit_box": step,
        "linear_profile_solves": 3 * (3 + 4 * width),
        "new_nonlinear_optimizations": 0,
        "predicted_steps_applied": False,
        "confidence_or_actual_precision_established": False,
    }
