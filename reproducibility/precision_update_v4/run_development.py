"""Apply the saved response once and seal every estimate before grading."""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.precision_update_v4.io import OUTPUT, PROTOCOL, verify, sha, write_new
from reproducibility.precision_update_v4.workflow import load_context, risk_for, model_record
from wphid.precision_update_v4.update import one_step


def main():
    manifest = verify()
    if (OUTPUT / 'estimate-seal.json').exists():
        print({'status': 'already_sealed', 'new_updates': 0}, flush=True)
        return
    if (OUTPUT / 'execution-started.json').exists():
        raise RuntimeError('Interrupted development: preserve records; no automatic budget reset')
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    write_new(OUTPUT / 'execution-started.json', {'at_UTC': datetime.now(timezone.utc).isoformat(),
              'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
              'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'), 'budget': PROTOCOL['budget']})
    records = {}
    for entry in manifest['points']:
        began_cpu, began_wall = time.process_time(), time.perf_counter()
        context = None
        try:
            if began_cpu-started_cpu >= PROTOCOL['budget']['process_cpu_seconds'] or began_wall-started_wall >= PROTOCOL['budget']['wall_seconds']:
                raise TimeoutError('Fixed total development budget exhausted')
            context = load_context(entry)
            control = model_record(entry, context, context.parameters,
                                   context.point['mean_only_conditional_scaled_coefficients'], 'mean_control',
                                   context.point['mean_only_quadratic_profile'], 0., 0.)
        except Exception as error:
            control = {**entry, 'status': 'failed_control_retained', 'kind': 'mean_control',
                       'error_type': type(error).__name__, 'error': str(error), 'completed': True}
        write_new(OUTPUT / 'controls' / (entry['id'] + '.json'), control)
        try:
            if context is None or control['status'] != 'success':
                raise ValueError('No valid fixed input/control; update not attempted')
            risk = risk_for(context)
            parameters, profile = one_step(risk, context.parameters, context.candidate.flat_bounds,
                                           context.point['predicted_internal_parameter_shift_in_unit_box'],
                                           PROTOCOL['radius_unit_box'])
            update = model_record(entry, context, parameters, profile.coefficients, 'update', profile.objective,
                                  time.process_time()-began_cpu, time.perf_counter()-began_wall)
            update['minimum_corrected_gram_relative_eigenvalue'] = min(profile.gram_margins)
        except Exception as error:
            update = {**entry, 'status': 'failed_update_retained', 'kind': 'update', 'completed': True,
                      'error_type': type(error).__name__, 'error': str(error),
                      'new_nonlinear_optimizations': 0, 'fallback_used': False}
        write_new(OUTPUT / 'updates' / (entry['id'] + '.json'), update)
        for directory in ('controls', 'updates'):
            name = directory + '/' + entry['id'] + '.json'
            records[name] = sha(OUTPUT / name)
        print({'point': entry['id'], 'control': control['status'], 'update': update['status'],
               'cpu_seconds_so_far': time.process_time()-started_cpu}, flush=True)
    verify()
    cpu, wall = time.process_time()-started_cpu, time.perf_counter()-started_wall
    write_new(OUTPUT / 'estimate-seal.json', {'version': PROTOCOL['version'],
              'sealed_at_UTC': datetime.now(timezone.utc).isoformat(), 'record_sha256': records,
              'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
              'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'), 'records': len(records),
              'cpu_seconds': cpu, 'wall_seconds': wall, 'new_nonlinear_optimizations': 0,
              'truth_read_during_estimation': False, 'new_inputs_or_noise': 0})
    print({'sealed_records': len(records), 'cpu_seconds': cpu, 'wall_seconds': wall}, flush=True)


if __name__ == '__main__':
    main()
