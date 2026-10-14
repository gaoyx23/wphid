"""Reconstruct frozen observations and predict without refitting old models."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from reproducibility.precision_update_v4.io import ROOT, PREFLIGHT, read
from reproducibility.final_candidate_v1.scientific_io import load_conditions
from wphid.final_candidate_v1.problem import TrainingScale, paper_families, validate_split
from wphid.final_candidate_v1.solver import Candidate, Mechanism, design
from wphid.precision_development_v3.moment_risk import WeakMomentRisk, weak_operators
from wphid.precision_update_v4.update import physical_units


def digest_array(values):
    array = np.asarray(values, dtype='<f8', order='C')
    payload = json.dumps(list(array.shape), separators=(',', ':')).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Context:
    point: dict
    family: object
    candidate: object
    scale: object
    train: object
    heldout: object
    operators: object
    parameters: np.ndarray


def load_context(entry):
    point = read(ROOT / entry['preflight_record'])
    protocol = read(ROOT / 'reproducibility/precision_development_v3_r1/preflight-protocol.json')
    if point['status'] != 'computed' or point['truth_read'] or point['predicted_steps_applied']:
        raise ValueError('Expected an unapplied truth-free completed preflight')
    source = point['input']
    if any(count for values in source['clipping'].values() for count in values.values()):
        raise ValueError('Acquisition clipping outside the declared noise model')
    conditions = load_conditions(ROOT / source['file'], source['sha256'])
    by_name = {c.name: c for c in conditions}
    validate_split(tuple(by_name), tuple(protocol['train_conditions']), tuple(protocol['heldout_conditions']))
    family = next(f for f in paper_families() if f.name == protocol['family'])
    training = tuple(by_name[n] for n in protocol['train_conditions'])
    heldout = tuple(by_name[n] for n in protocol['heldout_conditions'])
    scale = TrainingScale.fit(training, (family,))
    saved = read(ROOT / point['saved_fit'])['fit']
    if scale.fingerprint != point['training_scale_sha256'] or scale.fingerprint != saved['training_scale_sha256']:
        raise ValueError('Training scale changed')
    hypothesis = saved['hypothesis']
    if len(hypothesis) != 1 or hypothesis[0]['family'] != family.name or hypothesis[0]['groups'] != point['groups']:
        raise ValueError('Saved conditional hypothesis changed')
    parameters = np.asarray(hypothesis[0]['internal_parameters_by_group'], dtype=float).ravel()
    if not np.array_equal(parameters, point['saved_nonlinear_parameters']):
        raise ValueError('Initial parameters differ from the saved preflight')
    candidate = Candidate((Mechanism(hypothesis[0]['mechanism'], family, tuple(point['groups'])),))
    with np.load(ROOT / source['file'], allow_pickle=False) as arrays:
        operators = weak_operators(arrays['t'], arrays['z'], arrays['r'],
                                   tuple(protocol['window_shape']), tuple(protocol['stride_shape']))
    return Context(point, family, candidate, scale, scale.apply(training),
                   scale.apply(heldout), operators, parameters)


def risk_for(context):
    return WeakMomentRisk(context.train, context.family, tuple(context.point['groups']), context.operators,
                          context.point['sigma_estimates_from_training_observed_field_RMS'])


def predict(view, candidate, parameters, coefficients):
    matrices = design(view, candidate, parameters)
    predictions = tuple(matrix @ values for matrix, values in zip(matrices, coefficients))
    residuals = tuple(float(np.linalg.norm(p-y)/max(np.linalg.norm(y), 1e-30))
                      for p, y in zip(predictions, view.targets))
    rss = tuple(float((p-y) @ (p-y)) for p, y in zip(predictions, view.targets))
    return {'equation_weak_target_relative_residual': list(residuals),
            'equation_rss': list(rss), 'prediction_sha256': [digest_array(p) for p in predictions]}


def model_record(entry, context, parameters, coefficients, kind, risk_value, cpu, wall):
    parameters = np.asarray(parameters, dtype=float)
    coefficients = tuple(np.asarray(c, dtype=float) for c in coefficients)
    physical = physical_units(context.scale, context.family, coefficients)
    train = predict(context.train, context.candidate, parameters, coefficients)
    heldout = predict(context.heldout, context.candidate, parameters, coefficients)
    mechanism = context.candidate.mechanisms[0]
    return {**entry, 'completed': True, 'status': 'success', 'kind': kind,
            'flat_parameters': parameters.tolist(),
            'fit': {'hypothesis': [{'mechanism': mechanism.name, 'family': mechanism.family.name,
                                   'groups': list(mechanism.groups),
                                   'internal_parameters_by_group': parameters.reshape(-1, 3).tolist()}],
                    'scaled_coefficients': [c.tolist() for c in coefficients],
                    'physical_coefficients': [c.tolist() for c in physical],
                    'training_scale_sha256': context.scale.fingerprint,
                    'training_objective': float(risk_value),
                    'objective_identity': 'corrected_quadratic_risk' if kind == 'update' else 'saved_mean_only_quadratic_risk',
                    'equation_training_rss': train['equation_rss'],
                    'common_final_score': None, 'candidate_selection_performed': False,
                    'optimizer_status': 'fixed_one_step_no_optimizer' if kind == 'update' else 'reused_no_refit',
                    'cpu_seconds': float(cpu), 'wall_seconds': float(wall)},
            'raw_training_prediction': train, 'frozen_heldout': heldout,
            'truth_read_during_estimation': False, 'new_nonlinear_optimizations': 0,
            'new_linear_profiles': int(kind == 'update'), 'mean_control_refitted': False,
            'physical_trajectory_integrations': 0, 'old_results_changed': False}


def replay(entry, record):
    if record['status'] != 'success':
        return {'passed': False, 'reason': 'Retained failed estimate'}
    context = load_context(entry)
    parameters = np.asarray(record['flat_parameters'])
    coefficients = tuple(np.asarray(c) for c in record['fit']['scaled_coefficients'])
    expected = context.parameters.copy()
    if record['kind'] == 'update':
        bounds = np.asarray(context.candidate.flat_bounds)
        expected += (bounds[:, 1]-bounds[:, 0])*np.asarray(context.point['predicted_internal_parameter_shift_in_unit_box'])
    elif record['kind'] == 'mean_control':
        if any(not np.array_equal(c, s) for c, s in zip(coefficients, context.point['mean_only_conditional_scaled_coefficients'])):
            raise ValueError('Reused mean-only control coefficients changed')
    else:
        raise ValueError('Unknown saved model kind')
    if not np.array_equal(parameters, expected):
        raise ValueError('Saved fixed update changed')
    if record['fit']['training_scale_sha256'] != context.scale.fingerprint:
        raise ValueError('Replay scale mismatch')
    for actual, saved in zip(physical_units(context.scale, context.family, coefficients), record['fit']['physical_coefficients']):
        np.testing.assert_array_equal(actual, saved)
    for key, view in (('raw_training_prediction', context.train), ('frozen_heldout', context.heldout)):
        actual = predict(view, context.candidate, parameters, coefficients)
        if actual['prediction_sha256'] != record[key]['prediction_sha256']:
            raise ValueError('Frozen prediction mismatch')
        np.testing.assert_array_equal(actual['equation_weak_target_relative_residual'], record[key]['equation_weak_target_relative_residual'])
    return {'passed': True, 'new_coefficient_profiles': 0, 'new_nonlinear_optimizations': 0}


def paired_ratio(rows, reference, role):
    chosen = [r for r in rows if r['role'] == role]
    if not chosen or any(not r['assessable'] or not reference[r['id']]['assessable']
                         or reference[r['id']]['case_score'] <= 0 for r in chosen):
        return None
    return float(np.median([r['case_score']/reference[r['id']]['case_score'] for r in chosen]))
