"""One common nonnegative criterion for finite, possibly nested candidates."""

from dataclasses import dataclass
import hashlib
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares


class RankFailure(ValueError):
    pass


class BudgetFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Model:
    key: str
    bounds: tuple
    coefficient_count: int
    matrix: object

    @property
    def parameter_count(self):
        return self.coefficient_count + len(self.bounds)


class CommonCriterion:
    def __init__(self, target, row_ids, covariance=None, rank_tolerance=1e-12):
        self.target = np.array(target, dtype=float, copy=True)
        self.row_ids = tuple(row_ids)
        if (self.target.ndim != 1 or not len(self.target) or not np.isfinite(self.target).all()
                or len(self.row_ids) != len(self.target) or len(set(self.row_ids)) != len(self.row_ids)
                or not 0 < rank_tolerance < 1):
            raise ValueError('Invalid common moment endpoint')
        n = len(self.target)
        covariance = np.eye(n) if covariance is None else np.asarray(covariance, dtype=float)
        if covariance.shape != (n, n) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15):
            raise ValueError('Invalid fixed common covariance')
        # No clipping or parameter-dependent covariance update.
        cholesky = np.linalg.cholesky(covariance)
        self.whitener = solve_triangular(cholesky, np.eye(n), lower=True)
        self.rank_tolerance = rank_tolerance
        self.target.setflags(write=False)
        self.whitener.setflags(write=False)
        self.fingerprint = hashlib.sha256(repr(self.row_ids).encode() + self.target.tobytes()
                                           + self.whitener.tobytes()).hexdigest()

    def profile(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.target) or not np.isfinite(matrix).all():
            raise ValueError('Candidate differs from common moment rows')
        a, z = self.whitener @ matrix, self.whitener @ self.target
        if matrix.shape[1] == 0:
            coefficients, rank, singular = np.empty(0), 0, np.empty(0)
        else:
            # Column equilibration changes numerical coordinates, not the
            # common residual or coefficient constraints.
            norms = np.linalg.norm(a, axis=0)
            if np.any(norms <= 0):
                raise RankFailure('Zero candidate column')
            scaled, _, rank, singular = np.linalg.lstsq(a/norms, z, rcond=self.rank_tolerance)
            if rank != matrix.shape[1]:
                raise RankFailure('Candidate linear moment map is rank deficient')
            coefficients = scaled/norms
        residual = z - a @ coefficients
        return {'coefficients': coefficients, 'residual': residual,
                'objective': float(residual @ residual), 'linear_rank': int(rank),
                'linear_singular_values_equilibrated': singular}


def fit_model(criterion, model, *, starts=(.5, .25, .75), max_evaluations=600,
              cpu_seconds=60., wall_seconds=90.):
    begin_cpu, begin_wall = time.process_time(), time.perf_counter()
    bounds = np.asarray(model.bounds, dtype=float).reshape(-1, 2)
    if (not model.key or model.coefficient_count < 0 or not np.isfinite(bounds).all()
            or np.any(bounds[:, 1] <= bounds[:, 0]) or not starts
            or any(not 0 < s < 1 for s in starts) or max_evaluations < 1):
        raise ValueError('Invalid fixed optimization declaration')
    count, best = 0, None
    statuses = []

    def evaluate(unit):
        nonlocal count, best
        if count >= max_evaluations or time.process_time()-begin_cpu >= cpu_seconds or time.perf_counter()-begin_wall >= wall_seconds:
            raise BudgetFailure('Declared entire-fit budget exhausted')
        parameters = bounds[:, 0] + (bounds[:, 1]-bounds[:, 0])*np.asarray(unit)
        matrix = np.asarray(model.matrix(parameters))
        if matrix.shape != (len(criterion.target), model.coefficient_count):
            raise ValueError('Declared coefficient layout differs from moment map')
        count += 1
        profile = criterion.profile(matrix)
        if best is None or profile['objective'] < best['profile']['objective']:
            best = {'parameters': parameters.copy(), 'profile': profile}
        return profile['residual']

    status, error = 'completed', None
    try:
        if not len(bounds):
            evaluate(np.empty(0))
            statuses.append('linear_only')
        else:
            for start in starts:
                result = least_squares(evaluate, np.full(len(bounds), start), bounds=(0., 1.),
                                       xtol=1e-10, ftol=1e-10, gtol=1e-10, diff_step=1e-5,
                                       max_nfev=max_evaluations)
                statuses.append({'status': int(result.status), 'converged': bool(result.success)})
            if any(not s['converged'] for s in statuses):
                status = 'nonconverged'
    except BudgetFailure as failure:
        status, error = 'budget_exhausted', str(failure)
    except (ValueError, np.linalg.LinAlgError) as failure:
        status, error = 'failed_retained', str(failure)
    return {'model_key': model.key, 'status': status, 'error': error,
            'parameters': None if best is None else best['parameters'],
            'profile': None if best is None else best['profile'],
            'parameter_count': model.parameter_count, 'objective_evaluations': count,
            'starts': list(starts), 'restart_statuses': statuses,
            'cpu_seconds': time.process_time()-begin_cpu, 'wall_seconds': time.perf_counter()-begin_wall,
            'common_endpoint_sha256': criterion.fingerprint, 'truth_used': False}


def select_models(criterion, models, penalty, **budget):
    models = tuple(models)
    if not models or len({m.key for m in models}) != len(models) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid finite candidate declaration')
    records = [fit_model(criterion, model, **budget) for model in models]
    for record in records:
        record['common_score'] = (None if record['profile'] is None else
                                  record['profile']['objective'] + penalty*record['parameter_count'])
    eligible = [r for r in records if r['status'] == 'completed']
    selected = min(eligible, key=lambda r: (r['common_score'], r['model_key'])) if eligible else None
    return {'records': records, 'selected_key': None if selected is None else selected['model_key'],
            'all_declared_candidates_completed': len(eligible) == len(models), 'penalty': float(penalty),
            'criterion_identity': 'fixed_common_weighted_moment_square_plus_dimension_penalty',
            'likelihood_or_calibrated_selection_probability': False}


def tangent_diagnostics(criterion, model, parameters, unit_step=1e-5):
    parameters = np.asarray(parameters, dtype=float)
    bounds = np.asarray(model.bounds, dtype=float).reshape(-1, 2)
    if parameters.shape != (len(bounds),) or not np.isfinite(parameters).all() or not unit_step > 0:
        raise ValueError('Invalid tangent point')
    matrix = np.asarray(model.matrix(parameters))
    profile = criterion.profile(matrix)
    whitened = criterion.whitener @ matrix
    q, _ = np.linalg.qr(whitened, mode='reduced')
    columns, original_columns = [], []
    for index, (lower, upper) in enumerate(bounds):
        step = unit_step*(upper-lower)
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        if plus[index] >= upper or minus[index] <= lower:
            raise ValueError('Tangent requires a declared interior point')
        derivative = (model.matrix(plus)-model.matrix(minus))/(2*unit_step)
        column = criterion.whitener @ (derivative @ profile['coefficients'])
        original_columns.append(column)
        columns.append(column - q @ (q.T @ column))
    tangent = np.column_stack(columns) if columns else np.empty((len(criterion.target), 0))
    singular = np.linalg.svd(tangent, compute_uv=False)
    reference = np.linalg.norm(np.column_stack(original_columns), 2) if original_columns else 0.
    threshold = max(reference*criterion.rank_tolerance, np.finfo(float).eps*reference*len(criterion.target))
    rank = int(np.sum(singular > threshold))
    return {'rank': rank, 'dimension': len(bounds), 'unit_box_singular_values': singular,
            'minimum_to_maximum_ratio': float(singular[-1]/singular[0]) if len(singular) and singular[0] > 0 else None,
            'parameter_precision_established': False, 'tangent': tangent}
