"""Known-support first-order Gaussian weak PDE quasi-likelihood.

This independent comparator follows the residual covariance plus log determinant
idea in nonlinear WENDy. It does not execute or claim equivalence to WENDy.jl:
its residual is the archived axisymmetric PDE weak map, not an ODE time map.
Noise moments and first-order linearization are explicit approximations.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack, vstack


class GaussianWeakProblem:
    def __init__(self, training, family, groups, operators, sigma_by_condition):
        if family.name != 'arrhenius_oxygen_damage' or len(groups) != 3 or any(g is None for g in groups):
            raise ValueError('Three active known-support Arrhenius destinations required')
        if set(groups) != set(range(max(groups)+1)):
            raise ValueError('Partition must use consecutive groups')
        self.training, self.family, self.groups = training, family, tuple(groups)
        self.operators = tuple(csr_matrix(o) for o in operators)
        self.sigma = tuple(np.asarray(s, dtype=float) for s in sigma_by_condition)
        if len(self.operators) != 4 or len(self.sigma) != len(training.conditions) or any(
                s.shape != (3,) or not np.isfinite(s).all() or np.any(s <= 0) for s in self.sigma):
            raise ValueError('Four operators and three positive declared sigmas per condition required')
        self.group_count = max(groups)+1
        self.bounds = np.asarray(family.bounds*self.group_count, dtype=float)
        self.lower, self.span = self.bounds[:, 0], np.diff(self.bounds, axis=1).ravel()
        self.theta_count = len(self.lower)
        self.parameter_count = self.theta_count+12
        self.zero = csr_matrix(self.operators[2].shape)
        self.blocks = []
        for condition, sigma in zip(training.conditions, self.sigma):
            difference = condition.projection-self.operators[2]
            if difference.nnz and np.max(np.abs(difference.data)) > 2e-13:
                raise ValueError('Supplied weak reaction map differs from archived input')
            for e, state in enumerate(condition.states):
                base = np.column_stack([o@state for o in self.operators[:3]])
                if not np.allclose(base, condition.bases[e], rtol=2e-11, atol=2e-12):
                    raise ValueError('Supplied base map differs from archived input')
            t, o, d = condition.states
            features = np.column_stack((-1/t, np.log(o), np.log1p(-d)))
            slopes = (1/t**2, 1/o, -1/(1-d))
            self.blocks.append((condition, sigma, features, slopes))

    def decode(self, unit):
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.theta_count,) or not np.isfinite(unit).all() or np.any(unit < 0) or np.any(unit > 1):
            raise ValueError('Internal parameters outside declared unit box')
        return np.minimum(np.maximum(self.lower+self.span*unit, self.bounds[:, 0]), self.bounds[:, 1])

    def initialization(self, fraction):
        theta = self.decode(np.full(self.theta_count, fraction)).reshape(-1, 3)
        coefficients = []
        for e in range(3):
            matrix = np.column_stack((self.training.bases[e], self.training.atom(self.family, theta[self.groups[e]])[e]))
            coefficients.extend(np.linalg.lstsq(matrix, self.training.targets[e], rcond=1e-12)[0])
        return np.r_[np.full(self.theta_count, fraction), coefficients]

    def value_gradient(self, vector, *, return_blocks=False):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.parameter_count,) or not np.isfinite(vector).all():
            raise ValueError('Invalid joint parameter vector')
        parameters = self.decode(vector[:self.theta_count]).reshape(-1, 3)
        coefficients = vector[self.theta_count:].reshape(3, 4)
        gradient, value, diagnostics = np.zeros_like(vector), 0., []
        projection, target_operator = self.operators[2], self.operators[3]
        weights = self.training.scale.target_weights
        norms = self.training.scale.base_norms
        atom_norms = self.training.scale.norms_for(self.family)
        for condition, sigma, features, slopes in self.blocks:
            rates, state_gradients = [], []
            for p in parameters:
                rate = np.exp(features@p)
                rates.append(rate)
                state_gradients.append(tuple(rate*p[h]*slopes[h] for h in range(3)))
            equation_jacobians, residuals = [], []
            for e in range(3):
                c, g = coefficients[e], self.groups[e]
                bs, rs = weights[e]/norms[e], weights[e]/atom_norms[e]
                base = condition.bases[e]*bs
                atom = np.asarray(projection@rates[g])*rs
                residuals.append(base@c[:3]+atom*c[3]-condition.targets[e]*weights[e])
                jacobians = []
                for h in range(3):
                    j = projection.multiply(state_gradients[g][h])*rs*c[3]
                    if h == e:
                        j = j-target_operator*weights[e]
                        for k in range(3):
                            j = j+self.operators[k]*(bs[k]*c[k])
                    jacobians.append(j.tocsr())
                equation_jacobians.append(jacobians)
            r = np.concatenate(residuals)
            noise_factor = hstack([vstack([j[h] for j in equation_jacobians])*sigma[h] for h in range(3)], format='csr')
            covariance = (noise_factor@noise_factor.T).toarray()
            if return_blocks:
                diagnostics.append({'residual':r.copy(), 'noise_factor':noise_factor.copy(), 'covariance':covariance.copy()})
            factor = cho_factor(covariance, lower=True, check_finite=False)
            precision_r = cho_solve(factor, r, check_finite=False)
            value += .5*(float(r@precision_r)+2*np.log(np.diag(factor[0])).sum())
            precision = cho_solve(factor, np.eye(len(r)), check_finite=False)
            covariance_score = precision-np.outer(precision_r, precision_r)
            factor_score = np.asarray(noise_factor.T@covariance_score.T).T
            rows = projection.shape[0]
            def covariance_derivative(e, derivatives):
                total = 0.
                for h, derivative in enumerate(derivatives):
                    sl = slice(h*projection.shape[1], (h+1)*projection.shape[1])
                    total += float(derivative.multiply(factor_score[e*rows:(e+1)*rows, sl]).sum())*sigma[h]
                return total
            for e in range(3):
                c, g = coefficients[e], self.groups[e]
                bs, rs = weights[e]/norms[e], weights[e]/atom_norms[e]
                a = precision_r[e*rows:(e+1)*rows]
                for k in range(3):
                    derivatives = [self.zero]*3
                    derivatives[e] = self.operators[k]*bs[k]
                    gradient[self.theta_count+4*e+k] += float(a@(condition.bases[e][:, k]*bs[k]))+covariance_derivative(e, derivatives)
                derivatives = [projection.multiply(x)*rs for x in state_gradients[g]]
                gradient[self.theta_count+4*e+3] += float(a@(projection@rates[g]*rs))+covariance_derivative(e, derivatives)
                for k in range(3):
                    span = self.span[3*g+k]
                    drate = rates[g]*features[:, k]*span
                    derivatives = []
                    for h in range(3):
                        dg = state_gradients[g][h]*features[:, k]*span
                        if h == k:
                            dg = dg+rates[g]*slopes[h]*span
                        derivatives.append(projection.multiply(dg)*(rs*c[3]))
                    gradient[3*g+k] += float(a@(projection@drate*(rs*c[3])))+covariance_derivative(e, derivatives)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError('Nonfinite quasi-likelihood or gradient')
        return (float(value), gradient, diagnostics) if return_blocks else (float(value), gradient)

    def physical_coefficients(self, vector):
        coefficients = np.asarray(vector)[self.theta_count:].reshape(3, 4)
        result = []
        for e, c in enumerate(coefficients):
            result.append(np.r_[c[:3]/self.training.scale.base_norms[e], c[3]/self.training.scale.norms_for(self.family)[e]])
        return tuple(result)


def fit_gaussian_weak(problem, *, starts=(.5, .25, .75), cpu_seconds=60., max_evaluations=2000):
    if cpu_seconds <= 0 or max_evaluations < 1 or not starts or any(not 0<s<1 for s in starts):
        raise ValueError('Invalid fixed fitting declaration')
    begin_cpu, begin_wall = time.process_time(), time.perf_counter()
    evaluations, best, statuses = 0, None, []
    class BudgetExhausted(RuntimeError):
        pass
    def objective(vector):
        nonlocal evaluations, best
        if evaluations >= max_evaluations or time.process_time()-begin_cpu >= cpu_seconds:
            raise BudgetExhausted('Entire-fit budget exhausted')
        evaluations += 1
        value, gradient = problem.value_gradient(vector)
        if best is None or value < best['objective']:
            best = {'vector': np.array(vector, copy=True), 'objective': value}
        return value, gradient
    status, error = 'completed', None
    try:
        for start in starts:
            initial = problem.initialization(start)
            result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                              bounds=[(0., 1.)]*problem.theta_count+[(None, None)]*12,
                              options={'maxiter': max_evaluations, 'ftol':1e-12, 'gtol':1e-7, 'maxls':30})
            statuses.append({'start':start, 'success':bool(result.success), 'message':str(result.message)})
        if any(not s['success'] for s in statuses):
            status = 'nonconverged'
    except BudgetExhausted as failure:
        status, error = 'budget_exhausted', str(failure)
    except (ValueError, np.linalg.LinAlgError) as failure:
        status, error = 'failed_retained', str(failure)
    return {'algorithm':'conditional_gaussian_weak_quasi_likelihood', 'official_wendy':False,
            'status':status, 'error':error, 'restart_statuses':statuses,
            'parameters':None if best is None else problem.decode(best['vector'][:problem.theta_count]).tolist(),
            'physical_coefficients':None if best is None else [c.tolist() for c in problem.physical_coefficients(best['vector'])],
            'scaled_coefficients':None if best is None else best['vector'][problem.theta_count:].reshape(3,4).tolist(),
            'quasi_negative_log_likelihood':None if best is None else best['objective'],
            'objective_evaluations':evaluations, 'cpu_seconds':time.process_time()-begin_cpu,
            'wall_seconds':time.perf_counter()-begin_wall, 'starts':list(starts), 'truth_used':False,
            'covariance':'parameter_dependent_first_order_raw_shared_observation', 'diagonal_regularization':0.,
            'linear_coefficient_constraints':'signed', 'mean_correction':False, 'parameter_count':problem.parameter_count}
