"""V5 independent lock, stored-input context and saved-model conversion."""

import importlib.metadata
from pathlib import Path
import sys

import numpy as np

from reproducibility.observation_moments_v5.io import ROOT, read, sha, plain, check_runtime
from reproducibility.precision_update_v4.workflow import load_context, predict
from wphid.observation_moments_v5.weak_assembler import WeakMomentAssembler

PROTOCOL_PATH = Path(__file__).resolve().parent / 'development-protocol.json'
PROTOCOL = read(PROTOCOL_PATH)
OUTPUT = ROOT / PROTOCOL['output_root']


def verify():
    check_runtime()
    freeze = read(OUTPUT / 'source-freeze.json')
    if freeze['python_version'] != sys.version or freeze['python_executable'] != str(Path(sys.executable).resolve()):
        raise ValueError('Frozen V5 runtime changed')
    for name, version in freeze['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Frozen V5 dependency changed: ' + name)
    for name, expected in freeze['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    manifest = read(OUTPUT / 'input-manifest.json')
    if manifest['source_freeze_sha256'] != sha(OUTPUT / 'source-freeze.json'):
        raise ValueError('Input source-freeze mismatch')
    for name, expected in manifest['file_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen input/prior result changed: ' + name)
    return manifest


def assembler_for(context, corrected):
    return WeakMomentAssembler(context.train.conditions, context.operators,
                               context.point['sigma_estimates_from_training_observed_field_RMS'],
                               corrected=corrected, field_units=PROTOCOL['field_units'])


def record_fit(entry, context, assembler, result, variant):
    result = plain(result)
    record = {**entry, 'variant': variant, 'completed': True,
              'status': 'success' if result['status'] == 'completed' else 'retained_' + result['status'],
              'optimization_record': result, 'truth_read_during_estimation': False,
              'candidate_selection_performed': False, 'independent_confirmation': False,
              'new_inputs_or_noise': 0}
    if result['profile'] is None:
        return record
    parameters = np.asarray(result['parameters'])
    physical = assembler.physical_coefficients(context.candidate, result['profile']['coefficients'])
    norms = tuple(np.r_[context.scale.base_norms[e], context.scale.norms_for(context.family)[e]] for e in range(3))
    coefficients = tuple(c*n for c, n in zip(physical, norms))
    # Freeze an exactly reversible representation in the original weak scale.
    physical = tuple(c/n for c, n in zip(coefficients, norms))
    mechanism = context.candidate.mechanisms[0]
    record.update({'flat_parameters': parameters.tolist(),
                   'fit': {'hypothesis': [{'mechanism': mechanism.name, 'family': mechanism.family.name,
                                          'groups': list(mechanism.groups),
                                          'internal_parameters_by_group': parameters.reshape(-1, 3).tolist()}],
                           'scaled_coefficients': [c.tolist() for c in coefficients],
                           'physical_coefficients': [c.tolist() for c in physical],
                           'training_scale_sha256': context.scale.fingerprint,
                           'training_objective': result['profile']['objective'],
                           'objective_identity': 'common_moment_square_' + variant,
                           'common_final_score': None, 'optimizer_status': result['status'],
                           'cpu_seconds': result['cpu_seconds'], 'wall_seconds': result['wall_seconds']},
                   'common_anchor_declaration_sha256': assembler.declaration_sha256,
                   'raw_training_prediction': predict(context.train, context.candidate, parameters, coefficients),
                   'frozen_heldout': predict(context.heldout, context.candidate, parameters, coefficients)})
    return record


def replay(entry, record):
    if 'fit' not in record:
        return {'passed': False, 'reason': 'No saved decomposition; failed attempt retained'}
    context = load_context(entry)
    parameters = np.asarray(record['flat_parameters'])
    coefficients = tuple(np.asarray(c) for c in record['fit']['scaled_coefficients'])
    if record['fit']['training_scale_sha256'] != context.scale.fingerprint:
        raise ValueError('Saved V5 scale changed')
    for e, (scaled, physical) in enumerate(zip(coefficients, record['fit']['physical_coefficients'])):
        norms = np.r_[context.scale.base_norms[e], context.scale.norms_for(context.family)[e]]
        np.testing.assert_array_equal(scaled/norms, physical)
    for key, view in (('raw_training_prediction', context.train), ('frozen_heldout', context.heldout)):
        current = predict(view, context.candidate, parameters, coefficients)
        if current['prediction_sha256'] != record[key]['prediction_sha256']:
            raise ValueError('Saved V5 raw prediction changed')
        np.testing.assert_array_equal(current['equation_weak_target_relative_residual'], record[key]['equation_weak_target_relative_residual'])
    return {'passed': True, 'new_parameter_fits_or_coefficient_profiles': 0}
