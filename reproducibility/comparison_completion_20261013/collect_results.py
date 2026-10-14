"""Consolidate immutable comparison receipts; no estimator or truth imports."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper_comparison_completion_20261013'
ADAM = ROOT / 'artifacts/paper_matched_completion_v1'
GAUSSIAN = ROOT / 'artifacts/paper_wendy_comparison_v1/conditional_development_20261013_attempt01'
GAUSSIAN_FRESH = ROOT / 'artifacts/paper_wendy_comparison_v2/fresh_conditional_confirmation_20261013_attempt01'
DECLARATION = Path(__file__).with_name('selection-declaration.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(freeze):
    for key in ('source_sha256', 'old_source_sha256', 'old_result_and_input_sha256'):
        for relative, expected in freeze.get(key, {}).items():
            if sha(ROOT / relative) != expected:
                raise ValueError('Frozen source or old control changed: ' + relative)
    for job in freeze.get('jobs', []):
        for kind in ('input', 'control'):
            if sha(ROOT / job[kind + '_file']) != job[kind + '_sha256']:
                raise ValueError('Gaussian input/control changed')


def directional_blocks(assessment, direction):
    other = 'adam' if direction == 'wphid' else 'wphid'
    rows = []
    for block in assessment['blocks']:
        scores = block['scores']
        own, competing = scores[direction], scores[other]
        won = (block['both_assessable'] and own is not None and competing is not None
               and competing > 0 and own <= .9 * competing)
        rows.append({'seed': block['seed'], 'direction': direction, 'win': bool(won)})
    return rows


def main():
    destination = OUTPUT / 'completion-assessment.json'
    if destination.exists():
        raise FileExistsError('Final consolidation exists; do not overwrite')
    declared = read(DECLARATION)
    external = read(ADAM / 'external-selection-declaration-receipt.json')
    if sha(DECLARATION) not in json.dumps(external):
        raise ValueError('Selection declaration is not bound before fresh inputs')
    dev = read(ADAM / 'development-assessment.json')
    gaussian = read(GAUSSIAN / 'assessment.json')
    gaussian_entry = read(GAUSSIAN / 'independent-entry-decision.json')
    for folder in (ADAM, GAUSSIAN):
        verify_source(read(folder / 'source-freeze.json'))
    receipts = {}
    for folder, stage in ((ADAM, 'development-'), (GAUSSIAN, '')):
        assessment_path = folder / (stage + 'assessment.json')
        execution_path = folder / (stage + 'execution-complete.json')
        receipts[assessment_path.relative_to(ROOT).as_posix()] = sha(assessment_path)
        receipts[execution_path.relative_to(ROOT).as_posix()] = sha(execution_path)
        completion = read(execution_path)
        for relative, expected in completion['fit_sha256'].items():
            path = ROOT / relative if folder == ADAM else folder / 'fits' / (relative + '.json')
            if sha(path) != expected:
                raise ValueError('Saved scientific fit changed')
    fresh = None
    decisions, symmetric_claims = {}, {}
    if dev['computability_gate']['passed']:
        fresh = read(ADAM / 'fresh-assessment.json')
        completion = read(ADAM / 'fresh-execution-complete.json')
        for relative, expected in completion['fit_sha256'].items():
            if sha(ROOT / relative) != expected:
                raise ValueError('Fresh fit changed')
        receipts[(ADAM / 'fresh-assessment.json').relative_to(ROOT).as_posix()] = sha(ADAM / 'fresh-assessment.json')
        receipts[(ADAM / 'fresh-execution-complete.json').relative_to(ROOT).as_posix()] = sha(ADAM / 'fresh-execution-complete.json')
        threshold = declared['role_specific_practical_use']['minimum_precision_passes_for_provisional_task_use']
        for role, programs in fresh['role_summary'].items():
            qualified = [name for name, summary in programs.items() if summary['precision_passes'] >= threshold]
            decisions[role] = {'precision_passes': {name: s['precision_passes'] for name, s in programs.items()},
                               'qualified_for_provisional_conditional_role': qualified,
                               'decision': 'no_qualified_precise_recovery' if not qualified else ('prefer_' + qualified[0] if len(qualified) == 1 else 'both_qualified_compare_accuracy_and_cost')}
        from scipy.stats import binomtest
        for direction in ('wphid', 'adam'):
            wins = sum(row['win'] for row in directional_blocks(fresh, direction))
            other = 'adam' if direction == 'wphid' else 'wphid'
            rates = all(s[direction]['precision_passes'] >= s[other]['precision_passes'] for s in fresh['role_summary'].values())
            symmetric_claims[direction] = {'wins': wins, 'denominator': 18,
                'one_sided_binomial_p': float(binomtest(wins, 18, .5, alternative='greater').pvalue),
                'precision_rate_gate': rates, 'familywise_threshold': .025,
                'passed': wins >= 14 and rates}
    elif (ADAM / 'fresh-inputs.json').exists():
        raise ValueError('Fresh inputs generated despite failed development gate')
    gaussian_fresh = None
    if gaussian_entry['entered_fresh_Gaussian_comparison']:
        gaussian_fresh = read(GAUSSIAN_FRESH / 'assessment.json')
        verify_source(read(GAUSSIAN_FRESH / 'source-freeze.json'))
        completed = read(GAUSSIAN_FRESH / 'execution-complete.json')
        controls = read(GAUSSIAN_FRESH / 'control-fit-identities.json')
        for identifier, expected in completed['fit_sha256'].items():
            if sha(GAUSSIAN_FRESH / 'fits' / (identifier + '.json')) != expected:
                raise ValueError('Fresh Gaussian fit changed')
        for relative, expected in controls['fit_sha256'].items():
            if sha(ROOT / relative) != expected:
                raise ValueError('Three-program comparative control changed')
        for name in ('assessment.json', 'execution-complete.json', 'control-fit-identities.json'):
            receipts[(GAUSSIAN_FRESH / name).relative_to(ROOT).as_posix()] = sha(GAUSSIAN_FRESH / name)
    elif GAUSSIAN_FRESH.exists():
        raise ValueError('Gaussian fresh outputs exist despite failed entry gate')
    receipts[(GAUSSIAN / 'independent-entry-decision.json').relative_to(ROOT).as_posix()] = sha(GAUSSIAN / 'independent-entry-decision.json')
    rows = []
    for row in dev['rows']:
        rows.append({'stage': 'development', 'seed': row['seed'], 'role': row['role'], 'program': row['program'],
                     'assessable': row['assessable'], 'precision_pass': row['precision_pass'], 'case_score': row['case_score'],
                     'cpu_seconds': row.get('cpu_seconds'), 'status': row.get('optimizer_status', 'failed'), 'independent': False})
    for row in gaussian['rows']:
        rows.append({'stage': 'development', 'seed': row['seed'], 'role': row['role'], 'program': 'gaussian_quasi_likelihood',
                     'assessable': row['assessable'], 'precision_pass': row['precision_pass'], 'case_score': row['case_score'],
                     'cpu_seconds': row['cpu_seconds'], 'status': row['status'], 'independent': False})
    if fresh:
        for row in fresh['rows']:
            rows.append({'stage': 'fresh_conditional_comparison', 'seed': row['seed'], 'role': row['role'], 'program': row['program'],
                         'assessable': row['assessable'], 'precision_pass': row['precision_pass'], 'case_score': row['case_score'],
                         'cpu_seconds': row.get('cpu_seconds'), 'status': row.get('optimizer_status', 'failed'), 'independent': True})
    if gaussian_fresh:
        for row in gaussian_fresh['rows']:
            if row['program'] != 'conditional_Gaussian_weak_quasi_likelihood':
                continue
            rows.append({'stage': 'fresh_conditional_comparison', 'seed': row['seed'], 'role': row['role'], 'program': 'gaussian_quasi_likelihood',
                         'assessable': row['assessable'], 'precision_pass': row['precision_pass'], 'case_score': row['case_score'],
                         'cpu_seconds': row.get('cpu_seconds'), 'status': row.get('optimizer_status', 'failed'), 'independent': True})
    csv_path = OUTPUT / 'comparison-cases.csv'
    with csv_path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipts[csv_path.relative_to(ROOT).as_posix()] = sha(csv_path)
    report = {'version': 'COMPARISON_COMPLETION_CONSOLIDATED_20261013_V1',
        'selection_declaration_sha256': sha(DECLARATION), 'bound_receipts': receipts,
        'development_adam': dev['role_summary'], 'development_adam_computability': dev['computability_gate'],
        'development_gaussian': gaussian['summary'], 'development_gaussian_computability': gaussian['qualification_gate_pass'],
        'gaussian_independent_entry': gaussian_entry,
        'fresh_gaussian_directional_family': None if gaussian_fresh is None else gaussian_fresh['directional_family'],
        'fresh_conditional_comparison': None if fresh is None else fresh['role_summary'],
        'original_directional_advantage': None if fresh is None else fresh['formal_conditional_advantage'],
        'symmetric_predeclared_advantage': symmetric_claims, 'task_specific_decisions': decisions,
        'new_development_fits': 36, 'old_wphid_controls_reused_not_refit': 18,
        'new_fresh_fits': (0 if fresh is None else 108) + (0 if gaussian_fresh is None else 54), 'new_noise_packets': 0 if fresh is None else 54,
        'new_physical_integrations': 0, 'official_wendy_scientific_runs': 0,
        'official_wendy_status': 'PDE operator/noise adaptation and official dense runtime resource qualification missing; transparent supplemental comparator completed',
        'full_method_adoption': False, 'full_estimator_independent_engineering_confirmation': 'not_entered_no_adopted_complete_version',
        'original_eight_package_precision_count': '6/8', 'main_directory_adoption_or_writes': False,
        'no_completed_old_experiment_reruns': True}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(destination), 'rows': len(rows), 'fresh': fresh is not None}, ensure_ascii=False))


if __name__ == '__main__':
    main()
