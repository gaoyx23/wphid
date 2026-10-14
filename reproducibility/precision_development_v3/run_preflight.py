"""Evaluate a frozen no-optimization screen; preserve failures and stop."""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

from preflight_io import OUTPUT, PROTOCOL, ROOT, read, sha, verify_freeze, write_new

import numpy as np

sys.path.insert(1, str(ROOT / 'reproducibility/final_candidate_v1'))
from scientific_io import load_conditions
from wphid.final_candidate_v1.problem import TrainingScale, paper_families, validate_split
from wphid.precision_development_v3.moment_risk import WeakMomentRisk, saved_point_preflight, weak_operators


def main():
    manifest = verify_freeze()
    destination = OUTPUT / 'assessment.json'
    if destination.exists():
        print({'status': 'already_complete_read_existing', 'new_calculations': 0}, flush=True)
        return
    # An interrupted scientific screen is not silently extended with a new cap.
    if (OUTPUT / 'execution-started.json').exists():
        raise RuntimeError('Interrupted preflight: preserve partial records and audit; no automatic budget restart')
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    write_new(OUTPUT / 'execution-started.json', {
        'at_UTC': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
        'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
        'budget': PROTOCOL['budget'], 'new_nonlinear_fits': 0,
    })
    family = next(f for f in paper_families() if f.name == PROTOCOL['family'])
    records = []
    for point in manifest['points']:
        name = point['id']
        elapsed_cpu, elapsed_wall = time.process_time()-started_cpu, time.perf_counter()-started_wall
        if elapsed_cpu >= PROTOCOL['budget']['maximum_process_cpu_seconds'] or elapsed_wall >= PROTOCOL['budget']['maximum_wall_seconds']:
            result = {'status': 'preflight_budget_exhausted', 'linear_profile_solves': 0}
        else:
            try:
                source = point['input']
                if any(count for value in source['clipping'].values() for count in value.values()):
                    raise ValueError('Acquisition clipping requires a different noise model')
                conditions = load_conditions(ROOT / source['file'], source['sha256'])
                by_name = {c.name: c for c in conditions}
                validate_split(tuple(by_name), tuple(PROTOCOL['train_conditions']), tuple(PROTOCOL['heldout_conditions']))
                training_conditions = tuple(by_name[n] for n in PROTOCOL['train_conditions'])
                scale = TrainingScale.fit(training_conditions, (family,))
                saved = read(ROOT / point['saved_fit'])['fit']
                if scale.fingerprint != saved['training_scale_sha256']:
                    raise ValueError('Training scale differs from saved point')
                mechanism = saved['hypothesis'][0]
                if mechanism['family'] != family.name or len(saved['hypothesis']) != 1:
                    raise ValueError('Saved point outside declared single-reaction scope')
                groups = tuple(mechanism['groups'])
                parameters = np.asarray(mechanism['internal_parameters_by_group']).ravel()
                with np.load(ROOT / source['file'], allow_pickle=False) as arrays:
                    operators = weak_operators(arrays['t'], arrays['z'], arrays['r'],
                                               tuple(PROTOCOL['window_shape']), tuple(PROTOCOL['stride_shape']))
                sigma = tuple(tuple(PROTOCOL['noise_fraction'] * np.sqrt(np.mean(s*s)) for s in c.states)
                              for c in training_conditions)
                risk = WeakMomentRisk(scale.apply(training_conditions), family, groups, operators, sigma)
                result = saved_point_preflight(risk, parameters,
                                               tuple(b for _ in range(max(groups)+1) for b in family.bounds),
                                               PROTOCOL['finite_difference_step_unit_box'])
                result['sigma_estimates_from_training_observed_field_RMS'] = [list(s) for s in sigma]
                result['saved_nonlinear_parameters'] = parameters.tolist()
                result['training_scale_sha256'] = scale.fingerprint
                result['groups'] = list(groups)
            except Exception as error:
                result = {'status': 'failed_preflight_retained', 'error_type': type(error).__name__,
                          'error': str(error), 'linear_profile_solves': None}
        record = {**point, **result, 'truth_read': False, 'heldout_used': False,
                  'new_nonlinear_optimizations': 0, 'old_result_or_grade_changed': False}
        write_new(OUTPUT / 'points' / (name + '.json'), record)
        records.append(record)
        print({'point': name, 'status': record['status'],
               'max_predicted_amplitude_shift': record.get('maximum_predicted_amplitude_fractional_shift'),
               'cpu_seconds_so_far': time.process_time()-started_cpu}, flush=True)
    verify_freeze()
    gate = PROTOCOL['screen_gate']
    computed = [r for r in records if r['status'] == 'computed']
    material = [r for r in computed if r['role'] in gate['changed_roles']
                and r['local_response_within_declared_radius']
                and (r['maximum_predicted_amplitude_fractional_shift'] >= gate['material_amplitude_fractional_shift']
                     or r['maximum_predicted_unit_box_shift'] >= gate['material_internal_parameter_unit_box_shift'])]
    cpu, wall = time.process_time()-started_cpu, time.perf_counter()-started_wall
    resource = cpu <= PROTOCOL['budget']['maximum_process_cpu_seconds'] and wall <= PROTOCOL['budget']['maximum_wall_seconds']
    screen = len(computed) == PROTOCOL['saved_points'] and len(material) >= gate['minimum_material_points_of_12'] and resource
    assessment = {
        'version': PROTOCOL['version'], 'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
        'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
        'record_sha256': {f'points/{r["id"]}.json': sha(OUTPUT / 'points' / (r['id']+'.json')) for r in records},
        'records': records, 'saved_points': len(records), 'assessable_points': len(computed),
        'material_changed_role_points_of_12': len(material), 'material_point_ids': [r['id'] for r in material],
        'practical_screen_passed': screen, 'resource_gate_passed': resource,
        'cpu_seconds': cpu, 'wall_seconds': wall,
        'new_nonlinear_fits': 0, 'new_observations': 0, 'new_noise_draws': 0,
        'new_physical_trajectory_integrations': 0, 'nonlinear_parameters_reoptimized': False,
        'linear_profile_solves_completed_records': sum(r['linear_profile_solves'] for r in computed),
        'partial_solve_counts_available_for_failed_records': False,
        'tangent_projection_linear_solves_completed_records': sum(3*len(r['saved_nonlinear_parameters']) for r in computed),
        'local_response_linear_solves_completed_records': len(computed),
        'actual_parameter_precision_or_adoption_established': False,
        'original_eight_package_and_V1_V2_grades_changed': False,
        'next_action': 'Freeze a separate conditional development protocol; screen is not an adoption gate' if screen
                       else 'Stop this corrected-risk branch; do not launch nonlinear fits or scans',
    }
    write_new(destination, assessment)
    write_new(OUTPUT / 'execution-complete.json', {'assessment_sha256': sha(destination),
              'finished_at_UTC': datetime.now(timezone.utc).isoformat(), 'practical_screen_passed': screen})
    print({'assessable': len(computed), 'material_changed_points': len(material),
           'practical_screen_passed': screen, 'cpu_seconds': cpu, 'new_nonlinear_fits': 0}, flush=True)


if __name__ == '__main__':
    main()
