"""Sealed V5 precision assessment and prediction replay, including failures."""

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
from reproducibility.observation_moments_v5.development import OUTPUT, PROTOCOL, verify, replay
from reproducibility.precision_update_v4.workflow import paired_ratio


def main():
    manifest = verify()
    destination = OUTPUT / 'assessment.json'
    if destination.exists():
        if read(OUTPUT / 'execution-complete.json')['assessment_sha256'] != sha(destination):
            raise ValueError('Completed assessment identity mismatch')
        print({'status': 'already_complete', 'new_fits': 0}, flush=True)
        return
    begin_cpu, begin_wall = time.process_time(), time.perf_counter()
    seal = read(OUTPUT / 'estimate-seal.json')
    if seal['records'] != PROTOCOL['new_conditional_fits'] or seal['source_freeze_sha256'] != sha(OUTPUT / 'source-freeze.json') or seal['input_manifest_sha256'] != sha(OUTPUT / 'input-manifest.json'):
        raise ValueError('All fixed attempts must be sealed before grading')
    for name, expected in seal['record_sha256'].items():
        if sha(OUTPUT / name) != expected:
            raise ValueError('Sealed attempt changed: ' + name)
    sys.path.insert(1, str(ROOT / 'reproducibility/final_candidate_v1'))
    from reproducibility.final_candidate_v1.assess_confirmation import assess_fit, descriptive
    old = read(ROOT / 'artifacts/paper_final_candidate_v1/scientific_confirmation_20261011/assessment.json')
    old_rows = {f"{r['seed']}_{r['role']}": r for r in old['Q1']['rows'] if r['program'] == 'wphid'}
    rows, replays = [], []
    for entry in manifest['points']:
        for variant in PROTOCOL['variants_in_fixed_order']:
            record = read(OUTPUT / 'fits' / (entry['id'] + '_' + variant + '.json'))
            rows.append({**assess_fit(record, entry['role']), 'id': entry['id'], 'seed': entry['seed'],
                         'variant': variant, 'scientific_status': record['status']})
            try:
                result = replay(entry, record)
            except Exception as error:
                result = {'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
            replays.append({'id': entry['id'], 'variant': variant, **result})
    corrected = [r for r in rows if r['variant'] == 'corrected']
    raw = {r['id']: r for r in rows if r['variant'] == 'uncorrected'}
    summary = {role: {variant: descriptive([r for r in rows if r['role'] == role and r['variant'] == variant])
                      for variant in PROTOCOL['variants_in_fixed_order']} for role in PROTOCOL['roles']}
    ratios = {role: {'vs_uncorrected': paired_ratio(corrected, raw, role),
                     'vs_V1': paired_ratio(corrected, old_rows, role)} for role in ('partial', 'independent')}
    cpu = seal['cpu_seconds']+time.process_time()-begin_cpu
    wall = seal['wall_seconds']+time.perf_counter()-begin_wall
    thresholds = PROTOCOL['adoption_gate']
    gates = {'all_36_completed_and_assessable': all(r['scientific_status'] == 'success' and r['assessable'] for r in rows),
             'corrected_precision_by_role': all(summary[role]['corrected']['precision_passes'] >= thresholds['precision_passes_corrected_minimum'][role] for role in PROTOCOL['roles']),
             'all_36_replays': len(replays) == 36 and all(r['passed'] for r in replays),
             'resource': cpu <= PROTOCOL['budget']['cpu_seconds_all_fits'] and wall <= PROTOCOL['budget']['wall_seconds_all_fits']}
    for role in ('partial', 'independent'):
        for key, gate_name in (('vs_uncorrected', 'maximum_median_paired_case_score_ratio_vs_uncorrected'),
                               ('vs_V1', 'maximum_median_paired_case_score_ratio_vs_V1')):
            gates[role + '_' + key] = ratios[role][key] is not None and ratios[role][key] <= thresholds[gate_name][role]
    adopted = all(gates.values())
    verify()
    for name, expected in seal['record_sha256'].items():
        if sha(OUTPUT / name) != expected:
            raise ValueError('Sealed fit changed during assessment')
    assessment = {'version': PROTOCOL['version'], 'rows': rows, 'role_summary': summary,
                  'paired_case_score_ratios': ratios, 'saved_model_replays': replays, 'gates': gates,
                  'adoption_gate_passed': adopted, 'truth_read_only_after_seal': True,
                  'estimate_seal_sha256': sha(OUTPUT / 'estimate-seal.json'),
                  'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
                  'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
                  'cpu_seconds_estimation_plus_assessment': cpu, 'wall_seconds_active_processes': wall,
                  'new_conditional_fit_attempts': 36, 'reused_V1_grades': 18,
                  'new_inputs_or_noise': 0, 'independent_confirmation': False,
                  'old_results_changed_or_rescored': False, 'original_eight_package_precision_passes': 6,
                  'next_action': 'Separate full-method development decision eligible' if adopted
                                 else 'Stop this fixed V5 panel; no tuning, extra blocks, open confirmation or engineering confirmation'}
    write_new(destination, assessment)
    write_new(OUTPUT / 'execution-complete.json', {'finished_at_UTC': datetime.now(timezone.utc).isoformat(),
              'assessment_sha256': sha(destination), 'adoption_gate_passed': adopted})
    print({'precision_passes': {role: summary[role]['corrected']['precision_passes'] for role in PROTOCOL['roles']},
           'uncorrected_precision_passes': {role: summary[role]['uncorrected']['precision_passes'] for role in PROTOCOL['roles']},
           'ratios': ratios, 'replays_passed': sum(r['passed'] for r in replays),
           'adoption_gate_passed': adopted, 'cpu_seconds': cpu}, flush=True)


if __name__ == '__main__':
    main()
