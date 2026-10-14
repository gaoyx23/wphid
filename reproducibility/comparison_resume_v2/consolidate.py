"""Read completed frozen comparisons and disclose administrative recovery costs."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper_comparison_resume_v2/final_evidence'
PAIRED = ROOT / 'artifacts/paper_comparison_resume_v2/paired_confirmation'
GAUSSIAN = ROOT / 'artifacts/paper_comparison_resume_v1/gaussian_development'
GAUSSIAN_FRESH = ROOT / 'artifacts/paper_comparison_resume_v2/gaussian_fresh_confirmation'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


original = module('frozen_comparison_collector', ROOT / 'reproducibility/comparison_completion_20261013/collect_results.py')
read, sha = original.read, original.sha


def verify_source(freeze):
    """Fresh Gaussian jobs contain public inputs; controls are bound separately."""
    for key in ('source_sha256', 'old_source_sha256', 'old_result_and_input_sha256', 'receipt_sha256'):
        for relative, expected in freeze.get(key, {}).items():
            if sha(ROOT / relative) != expected:
                raise ValueError('Frozen evidence changed: ' + relative)
    for job in freeze.get('jobs', []):
        for kind in ('input', 'control'):
            path_key, hash_key = kind + '_file', kind + '_sha256'
            if (path_key in job) != (hash_key in job):
                raise ValueError('Incomplete input/control identity')
            if path_key in job and sha(ROOT / job[path_key]) != job[hash_key]:
                raise ValueError('Scientific input or control changed')


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Unexpected workspace')
    if OUTPUT.exists():
        raise FileExistsError('Final evidence root exists; never replace')
    if not read(GAUSSIAN / 'independent-entry-decision.json')['entered_fresh_Gaussian_comparison']:
        raise ValueError('Expected predeclared development gate passed')
    required = [PAIRED / 'fresh-assessment.json', PAIRED / 'fresh-execution-complete.json',
                GAUSSIAN_FRESH / 'assessment.json', GAUSSIAN_FRESH / 'execution-complete.json',
                GAUSSIAN_FRESH / 'control-fit-identities.json', GAUSSIAN / 'saved-fit-replay-audit.json',
                GAUSSIAN / 'administrative-cost-audit.json', PAIRED / 'recovery-v2-plan.json',
                PAIRED / 'saved-point-replay-r1.json', PAIRED / 'administrative-assessment-context-r1.json',
                PAIRED / 'final-integrity-audit.json',
                GAUSSIAN_FRESH / 'saved-fit-replay-audit.json', GAUSSIAN_FRESH / 'administrative-cost-audit.json',
                GAUSSIAN_FRESH / 'all-stages-complete.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('Dependent evidence is incomplete: ' + str(path))
    if read(GAUSSIAN / 'saved-fit-replay-audit.json')['passed'] != 18:
        raise ValueError('Gaussian development saved-vector audit not complete')
    paired_completion = read(PAIRED / 'fresh-execution-complete.json')
    if paired_completion['declared_scored_jobs'] != 108:
        raise ValueError('Paired denominator changed')
    paired_replay = read(PAIRED / 'saved-point-replay-r1.json')
    verify_source(read(PAIRED / 'replay-r1-software/qualification.json'))
    if (paired_replay['replayed_successes'] + paired_replay['saved_failures_retained'] != 108
            or len(paired_replay['rows']) != 108
            or {row['id'] for row in paired_replay['rows']} != {Path(path).stem for path in paired_completion['fit_sha256']}
            or paired_replay['original_scientific_source_freeze_sha256'] != sha(PAIRED / 'recovery-v2-source-freeze.json')
            or paired_replay['qualification_sha256'] != sha(PAIRED / 'replay-r1-software/qualification.json')):
        raise ValueError('Paired saved points not all replayed or retained')
    paired_context = read(PAIRED / 'administrative-assessment-context-r1.json')
    if (paired_context['fresh_assessment_sha256'] != sha(PAIRED / 'fresh-assessment.json')
            or paired_context['replay_sha256'] != sha(PAIRED / 'saved-point-replay-r1.json')
            or paired_context['source_freeze_sha256'] != sha(PAIRED / 'recovery-v2-source-freeze.json')):
        raise ValueError('Paired recovery audit is not bound to final endpoints')
    for relative, expected in paired_replay['fit_sha256'].items():
        if paired_completion['fit_sha256'].get(relative) != expected or sha(ROOT / relative) != expected:
            raise ValueError('Paired replay endpoint identity differs')
    paired_integrity = read(PAIRED / 'final-integrity-audit.json')
    for key, name in (('source_freeze_sha256', 'recovery-v2-source-freeze.json'), ('assessment_sha256', 'fresh-assessment.json'),
                      ('replay_sha256', 'saved-point-replay-r1.json'), ('context_sha256', 'administrative-assessment-context-r1.json')):
        if paired_integrity[key] != sha(PAIRED / name):
            raise ValueError('Paired final integrity receipt is not bound')
    if paired_integrity['gitattributes_sha256'] != sha(ROOT / '.gitattributes'):
        raise ValueError('Attributes byte preservation failed')
    if read(GAUSSIAN_FRESH / 'execution-complete.json')['new_Gaussian_fits'] != 54:
        raise ValueError('Gaussian confirmation denominator changed')
    gaussian_replay = read(GAUSSIAN_FRESH / 'saved-fit-replay-audit.json')
    failures = sum(row.get('saved_failure_retained', False) for row in gaussian_replay['rows'])
    gaussian_freeze = read(GAUSSIAN_FRESH / 'source-freeze.json')
    if (len(gaussian_replay['rows']) != 54 or gaussian_replay['passed'] + failures != 54
            or {row['id'] for row in gaussian_replay['rows']} != {job['id'] for job in gaussian_freeze['jobs']}
            or gaussian_replay['source_freeze_sha256'] != sha(GAUSSIAN_FRESH / 'source-freeze.json')
            or gaussian_replay['execution_complete_sha256'] != sha(GAUSSIAN_FRESH / 'execution-complete.json')):
        raise ValueError('Gaussian confirmation saved vectors have replay mismatches')
    stages = read(GAUSSIAN_FRESH / 'all-stages-complete.json')
    for key, name in (('execution_sha256', 'execution-complete.json'), ('assessment_sha256', 'assessment.json'),
                      ('replay_sha256', 'saved-fit-replay-audit.json'), ('cost_sha256', 'administrative-cost-audit.json')):
        if stages[key] != sha(GAUSSIAN_FRESH / name):
            raise ValueError('Gaussian all-stage completion is not bound to final receipts')
    for path in [PAIRED / 'recovery-source-freeze.json', PAIRED / 'recovery-v2-source-freeze.json',
                 GAUSSIAN / 'recovery-source-freeze.json', GAUSSIAN_FRESH / 'source-freeze.json']:
        freeze = read(path)
        verify_source(freeze)
        for key in ('original_receipt_sha256', 'copied_fit_sha256', 'old_control_and_fit_sha256'):
            for relative, expected in freeze.get(key, {}).items():
                if sha(ROOT / relative) != expected:
                    raise ValueError('Frozen reused evidence changed: ' + relative)
    OUTPUT.mkdir(parents=True)
    original.OUTPUT, original.ADAM = OUTPUT, PAIRED
    original.GAUSSIAN, original.GAUSSIAN_FRESH = GAUSSIAN, GAUSSIAN_FRESH
    original.verify_source = verify_source
    original.main()
    base_path = OUTPUT / 'completion-assessment.json'
    base = read(base_path)
    plan = read(PAIRED / 'recovery-v2-plan.json')
    gaussian_cost = read(GAUSSIAN / 'administrative-cost-audit.json')
    gaussian_fresh_cost = read(GAUSSIAN_FRESH / 'administrative-cost-audit.json')
    cost = {
        'comparison_selected_complete_attempts': 18 + 18 + 108 + 54,
        'archived_WPHID_development_controls_reused': 18,
        'new_endpoint_fits': 198,
        'unique_final_confirmation_input_packets': 54,
        'paired_second_recovery_byte_reuses': 105,
        'paired_second_recovery_new_retained_attempts': 3,
        'first_paired_interruption_lost_nominal_interval_seconds': [0, 2400],
        'first_Gaussian_interruption_lost_nominal_interval_seconds': [0, 1200],
        'second_paired_interruption': plan['lost_CPU'],
        'actual_lost_work_has_unobserved_gap_or_between_call_overshoot': True,
        'candidate_cumulative_600_CPU_budget_preserved_claim': False,
        'fixed_retained_attempt_nominal_CPU_cap_seconds': 600,
        'one_complete_scored_attempt_per_job': True,
        'lost_models_unavailable_and_not_selected': True,
        'administrative_interruption_is_algorithm_failure': False,
        'Gaussian_development_selected_solver_CPU_seconds': gaussian_cost['selected_saved_solver_cpu_seconds'],
        'Gaussian_confirmation_selected_solver_CPU_seconds': gaussian_fresh_cost['selected_saved_solver_cpu_seconds'],
        'Gaussian_confirmation_preparation_CPU_seconds': gaussian_fresh_cost['preparation_cpu_seconds'],
        'paired_selected_worker_CPU_seconds': paired_completion['worker_job_cpu_seconds'],
        'solver_timing_is_full_end_to_end_minimum_cost_claim': False,
        'WPHID_and_ADAM_objectives_identical': False,
    }
    save(OUTPUT / 'administrative-cost-and-scope.json', cost)
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in required + [base_path, OUTPUT / 'administrative-cost-and-scope.json']}
    report = {
        'version': 'COMPARISON_COMPLETION_RECOVERED_V2_20261013',
        'completed_scientific_assessment': base,
        'administrative_cost_and_scope': cost,
        'bound_evidence_sha256': bindings,
        'consolidator_source_sha256': sha(Path(__file__)),
        'original_collector_source_sha256': sha(ROOT / 'reproducibility/comparison_completion_20261013/collect_results.py'),
        'gitattributes_sha256': sha(ROOT / '.gitattributes'),
        'all_required_scientific_comparisons_complete': True,
        'official_WENDy_runtime_comparison_complete': False,
        'official_WENDy_scientific_runs': 0,
        'full_open_estimator_adopted': False,
        'base_task_specific_decisions_scope': 'Predeclared WPHID/ADAM conditional role decisions only; not three-program rankings',
        'scientific_precision_scope': 'Same-fit reaction nonlinear parameters and active amplitudes; not all base physical coefficients',
        'main_directory_writes': False,
        'new_fits_or_noise_from_consolidation': 0,
    }
    save(OUTPUT / 'recovered-completion-assessment.json', report)
    plot = module('frozen_comparison_plotter', ROOT / 'reproducibility/comparison_completion_20261013/plot_results.py')
    plot.OUTPUT = OUTPUT
    plot.main()
    print(json.dumps({'complete': True, 'final_evidence': str(OUTPUT), 'new_fits_from_postprocessing': 0}))


if __name__ == '__main__':
    main()
