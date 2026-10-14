"""Single no-optimization numerical eligibility check before V5 fitting."""

import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.observation_moments_v5.io import read, write_new, sha, plain
from reproducibility.observation_moments_v5.development import OUTPUT, PROTOCOL, verify, assembler_for, load_context
from wphid.observation_moments_v5.estimator import tangent_diagnostics


def main():
    manifest = verify()
    destination = OUTPUT / 'numerical-preflight.json'
    if destination.exists():
        print({'status': 'already_complete', 'new_calculations': 0}, flush=True)
        return
    if (OUTPUT / 'preflight-started.json').exists():
        raise RuntimeError('Interrupted preflight; preserve and audit, no automatic retry')
    write_new(OUTPUT / 'preflight-started.json', {'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json')})
    begin_cpu, begin_wall = time.process_time(), time.perf_counter()
    rows = []
    for entry in [e for e in manifest['points'] if e['seed'] == PROTOCOL['fixed_seeds'][0]]:
        try:
            context = load_context(entry)
            assembler = assembler_for(context, True)
            model, criterion = assembler.model(context.candidate), assembler.criterion()
            diagnostic = tangent_diagnostics(criterion, model, context.parameters)
            diagnostic.pop('tangent')
            eligible = diagnostic['rank'] == diagnostic['dimension'] and diagnostic['minimum_to_maximum_ratio'] >= PROTOCOL['preflight']['minimum_unit_box_tangent_singular_value_ratio']
            rows.append({**entry, 'status': 'computed', 'eligible': bool(eligible), **plain(diagnostic),
                         'moment_dimension': len(criterion.target), 'linear_coefficients': model.coefficient_count,
                         'anchor_declaration_sha256': assembler.declaration_sha256})
            del assembler, model, criterion, context
        except Exception as error:
            rows.append({**entry, 'status': 'failed_retained', 'eligible': False, 'error_type': type(error).__name__, 'error': str(error)})
        print({k: rows[-1].get(k) for k in ('id', 'status', 'rank', 'minimum_to_maximum_ratio', 'eligible')}, flush=True)
    cpu, wall = time.process_time()-begin_cpu, time.perf_counter()-begin_wall
    passed = len(rows) == 3 and all(r['eligible'] for r in rows) and cpu <= PROTOCOL['preflight']['cpu_seconds'] and wall <= PROTOCOL['preflight']['wall_seconds']
    verify()
    write_new(destination, {'version': PROTOCOL['version'], 'points': rows, 'numerical_eligibility_passed': passed,
              'cpu_seconds': cpu, 'wall_seconds': wall, 'new_nonlinear_fits': 0, 'truth_read': False,
              'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
              'actual_parameter_precision_or_adoption_established': False})
    print({'numerical_eligibility_passed': passed, 'cpu_seconds': cpu}, flush=True)


if __name__ == '__main__':
    main()
