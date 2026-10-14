"""Grade only sealed new decompositions and replay without optimization."""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.precision_update_v4.io import ROOT, OUTPUT, PROTOCOL, read, sha, verify, write_new
from reproducibility.precision_update_v4.workflow import replay, paired_ratio


def main():
    manifest = verify()
    destination = OUTPUT / 'assessment.json'
    if destination.exists():
        if read(OUTPUT / 'execution-complete.json')['assessment_sha256'] != sha(destination):
            raise ValueError('Completed assessment identity mismatch')
        print({'status': 'already_completed', 'new_updates': 0}, flush=True)
        return
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    seal_path = OUTPUT / 'estimate-seal.json'
    seal = read(seal_path)
    if seal['records'] != 36 or seal['source_freeze_sha256'] != sha(OUTPUT / 'source-freeze.json'):
        raise ValueError('All 36 estimates must be sealed before truth grading')
    if seal['input_manifest_sha256'] != sha(OUTPUT / 'input-manifest.json'):
        raise ValueError('Sealed input manifest mismatch')
    for name, expected in seal['record_sha256'].items():
        if sha(OUTPUT / name) != expected:
            raise ValueError('Sealed estimate changed: ' + name)
    # This is the first import that permits a reference-truth read. Its original
    # grade function is reused; original assessment.main() is never invoked.
    sys.path.insert(1, str(ROOT / 'reproducibility/final_candidate_v1'))
    from reproducibility.final_candidate_v1.assess_confirmation import assess_fit, descriptive
    old = read(ROOT / 'artifacts/paper_final_candidate_v1/scientific_confirmation_20261011/assessment.json')
    old_rows = {f"{r['seed']}_{r['role']}": r for r in old['Q1']['rows'] if r['program'] == 'wphid'}
    updates, controls, replays = [], [], []
    for entry in manifest['points']:
        for directory, target in (('updates', updates), ('controls', controls)):
            record = read(OUTPUT / directory / (entry['id'] + '.json'))
            row = {**assess_fit(record, entry['role']), 'id': entry['id'], 'seed': entry['seed'],
                   'program': 'V4_update' if directory == 'updates' else 'reused_mean_control'}
            target.append(row)
            try:
                result = replay(entry, record)
            except Exception as error:
                result = {'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
            replays.append({'id': entry['id'], 'kind': record['kind'], **result})
    control_by_id = {r['id']: r for r in controls}
    roles = PROTOCOL['roles']
    summary = {role: {'updated': descriptive([r for r in updates if r['role'] == role]),
                      'mean_control': descriptive([r for r in controls if r['role'] == role]),
                      'V1_reused': descriptive([old_rows[e['id']] for e in manifest['points'] if e['role'] == role])}
               for role in roles}
    ratios = {role: {'vs_mean_only': paired_ratio(updates, control_by_id, role),
                     'vs_V1': paired_ratio(updates, old_rows, role)} for role in ('partial', 'independent')}
    cpu = seal['cpu_seconds'] + time.process_time()-started_cpu
    wall = seal['wall_seconds'] + time.perf_counter()-started_wall
    thresholds = PROTOCOL['adoption_gate']
    gates = {'all_36_assessable': all(r['assessable'] for r in updates+controls),
             'precision_by_role': all(summary[role]['updated']['precision_passes'] >= thresholds['updated_precision_pass_minimum'][role] for role in roles),
             'all_36_replays_pass': len(replays) == 36 and all(r['passed'] for r in replays),
             'resource': cpu <= PROTOCOL['budget']['process_cpu_seconds'] and wall <= PROTOCOL['budget']['wall_seconds']}
    for role in ('partial', 'independent'):
        for key, name in (('vs_mean_only', 'maximum_median_paired_case_score_ratio_vs_mean_only'),
                          ('vs_V1', 'maximum_median_paired_case_score_ratio_vs_V1')):
            gates[role + '_' + key] = ratios[role][key] is not None and ratios[role][key] <= thresholds[name][role]
    adopted = all(gates.values())
    assessment = {'version': PROTOCOL['version'], 'estimate_seal_sha256': sha(seal_path),
                  'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
                  'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
                  'updated_rows': updates, 'mean_control_rows': controls, 'role_summary': summary,
                  'paired_case_score_ratios': ratios, 'saved_model_replays': replays,
                  'gates': gates, 'adoption_gate_passed': adopted,
                  'cpu_seconds_estimation_plus_assessment': cpu, 'wall_seconds_active_processes': wall,
                  'new_parameter_updates': 18, 'reused_mean_controls': 18, 'reused_V1_grades': 18,
                  'new_nonlinear_optimizations': 0, 'new_linear_profiles_successful': sum(r['success'] for r in updates),
                  'new_linear_solves_successful_profiles': 3*sum(r['success'] for r in updates),
                  'new_inputs_or_noise': 0, 'new_physical_trajectory_integrations': 0,
                  'truth_read_only_after_estimate_seal': True, 'independent_confirmation': False,
                  'original_eight_package_precision_passes': 6, 'original_eight_package_n': 8,
                  'prior_results_rescored_or_changed': False,
                  'next_action': 'Eligible for a separately frozen next development decision' if adopted
                                 else 'Stop this fixed one-step branch; no repeated step, radius adjustment, extra blocks or independent confirmation'}
    verify()
    for name, expected in seal['record_sha256'].items():
        if sha(OUTPUT / name) != expected:
            raise ValueError('Estimate changed during replay or grading')
    write_new(destination, assessment)
    write_new(OUTPUT / 'execution-complete.json', {'finished_at_UTC': datetime.now(timezone.utc).isoformat(),
              'assessment_sha256': sha(destination), 'adoption_gate_passed': adopted})
    print({'precision_passes': {role: summary[role]['updated']['precision_passes'] for role in roles},
           'ratios': ratios, 'replays_passed': sum(r['passed'] for r in replays),
           'adoption_gate_passed': adopted, 'cpu_seconds': cpu}, flush=True)


if __name__ == '__main__':
    main()
