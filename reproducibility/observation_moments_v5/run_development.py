"""A fixed paired V5 batch on viewed inputs; no truth or adaptive extension."""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.observation_moments_v5.io import read, write_new, sha
from reproducibility.observation_moments_v5.development import OUTPUT, PROTOCOL, verify, assembler_for, load_context, record_fit
from wphid.observation_moments_v5.estimator import fit_model


def main():
    manifest = verify()
    if (OUTPUT / 'estimate-seal.json').exists():
        print({'status': 'already_sealed', 'new_fits': 0}, flush=True)
        return
    preflight = read(OUTPUT / 'numerical-preflight.json')
    if not preflight['numerical_eligibility_passed'] or preflight['input_manifest_sha256'] != sha(OUTPUT / 'input-manifest.json'):
        raise ValueError('Numerical eligibility did not pass; do not fit')
    if (OUTPUT / 'execution-started.json').exists():
        raise RuntimeError('Interrupted batch; preserve partial outputs, no automatic budget reset')
    write_new(OUTPUT / 'execution-started.json', {'at_UTC': datetime.now(timezone.utc).isoformat(),
              'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
              'numerical_preflight_sha256': sha(OUTPUT / 'numerical-preflight.json'), 'budget': PROTOCOL['budget']})
    begin_cpu, begin_wall = time.process_time(), time.perf_counter()
    identities = {}
    optimization = PROTOCOL['optimization']
    budget = PROTOCOL['budget']
    for entry in manifest['points']:
        for variant in PROTOCOL['variants_in_fixed_order']:
            try:
                if time.process_time()-begin_cpu >= budget['cpu_seconds_all_fits'] or time.perf_counter()-begin_wall >= budget['wall_seconds_all_fits']:
                    raise TimeoutError('Entire fixed batch budget exhausted; attempt retained')
                context = load_context(entry)
                assembler = assembler_for(context, variant == 'corrected')
                model, criterion = assembler.model(context.candidate), assembler.criterion()
                cpu_left = budget['cpu_seconds_all_fits']-(time.process_time()-begin_cpu)
                wall_left = budget['wall_seconds_all_fits']-(time.perf_counter()-begin_wall)
                result = fit_model(criterion, model, starts=tuple(optimization['starts_unit_box']),
                                   max_evaluations=optimization['maximum_objective_evaluations_all_starts'],
                                   cpu_seconds=min(optimization['cpu_seconds_per_fit'], max(cpu_left, 0.)),
                                   wall_seconds=min(optimization['wall_seconds_per_fit'], max(wall_left, 0.)))
                record = record_fit(entry, context, assembler, result, variant)
                del context, assembler, model, criterion, result
            except Exception as error:
                record = {**entry, 'variant': variant, 'completed': True, 'status': 'failed_retained',
                          'error_type': type(error).__name__, 'error': str(error), 'truth_read_during_estimation': False}
            relative = f"fits/{entry['id']}_{variant}.json"
            write_new(OUTPUT / relative, record)
            identities[relative] = sha(OUTPUT / relative)
            print({'id': entry['id'], 'variant': variant, 'status': record['status'],
                   'evaluations': record.get('optimization_record', {}).get('objective_evaluations'),
                   'cpu_seconds_so_far': time.process_time()-begin_cpu}, flush=True)
    verify()
    write_new(OUTPUT / 'estimate-seal.json', {'version': PROTOCOL['version'], 'records': len(identities),
              'record_sha256': identities, 'sealed_at_UTC': datetime.now(timezone.utc).isoformat(),
              'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
              'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
              'cpu_seconds': time.process_time()-begin_cpu, 'wall_seconds': time.perf_counter()-begin_wall,
              'truth_read_during_estimation': False, 'new_inputs_or_noise': 0})
    print({'sealed_records': len(identities), 'cpu_seconds': time.process_time()-begin_cpu}, flush=True)


if __name__ == '__main__':
    main()
