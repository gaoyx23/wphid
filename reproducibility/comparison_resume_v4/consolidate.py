"""Complete the declared study with exact byte reuse after an orderly pause."""
import importlib.util
import csv
import json
from statistics import median
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper_comparison_resume_v4/final_evidence'
FRESH = ROOT / 'artifacts/paper_comparison_resume_v4/gaussian_fresh_confirmation'
PAUSE = ROOT / 'artifacts/paper_comparison_pause_20261014/paused-state.json'
spec = importlib.util.spec_from_file_location('comparison_recovered_v2', ROOT / 'reproducibility/comparison_resume_v2/consolidate.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def verify_resume():
    frozen = c.read(FRESH / 'recovery-source-freeze.json')
    for section in ('source_sha256', 'receipt_and_old_fit_sha256', 'copied_fit_sha256'):
        for relative, expected in frozen[section].items():
            if c.sha(ROOT / relative) != expected:
                raise ValueError('Frozen orderly resume source or endpoint changed: ' + relative)
    if (frozen['source_freeze_sha256'] != c.sha(FRESH / 'source-freeze.json')
            or frozen['completed_byte_reuses'] != 20 or frozen['new_never_started_jobs'] != 34
            or frozen['unique_endpoint_denominator'] != 54 or frozen['new_fits_at_freeze'] != 0):
        raise ValueError('Resume denominator or original source changed')
    qualification = ROOT / 'reproducibility/wendy_comparison_resume_v4/qualification.json'
    if frozen['qualification_sha256'] != c.sha(qualification):
        raise ValueError('New administrative qualification identity changed')
    c.verify_source(c.read(qualification))
    pause = c.read(PAUSE)
    all_jobs = c.read(FRESH / 'source-freeze.json')['jobs']
    copied_ids = {Path(relative).stem for relative in frozen['copied_fit_sha256']}
    paused_ids = {Path(relative).stem for relative in pause['fit_sha256']}
    if (copied_ids != paused_ids or frozen['pending_jobs'] != pause['remaining_jobs']
            or {job['id'] for job in frozen['pending_jobs']} != {j['id'] for j in all_jobs} - copied_ids):
        raise ValueError('Resume did not reuse20 and run only their exact34 complement')
    completion = c.read(FRESH / 'execution-complete.json')
    context = c.read(FRESH / 'administrative-resume-context.json')
    stages = c.read(FRESH / 'all-stages-complete.json')
    if (completion['recovery_source_freeze_sha256'] != c.sha(FRESH / 'recovery-source-freeze.json')
            or context['recovery_source_freeze_sha256'] != c.sha(FRESH / 'recovery-source-freeze.json')
            or stages['administrative_context_sha256'] != c.sha(FRESH / 'administrative-resume-context.json')
            or context['cost_sha256'] != c.sha(FRESH / 'administrative-cost-audit.json')):
        raise ValueError('New all-stage resume evidence is not bound')
    for evidence in (completion, context, stages):
        if evidence['completed_byte_reuses'] != 20 or evidence['new_retained_resume_fits'] != 34:
            raise ValueError('New resume counts changed')
    if context['lost_model_or_CPU_at_orderly_pause'] != 0 or not context['original20_endpoint_bytes_unchanged']:
        raise ValueError('Orderly pause was not lossless')
    return context


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Unexpected actual workspace')
    if OUTPUT.exists():
        raise FileExistsError('Completed consolidation must never be replaced')
    context = verify_resume()
    c.OUTPUT, c.GAUSSIAN_FRESH = OUTPUT, FRESH
    c.main()
    with (OUTPUT / 'comparison-cases.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    identities = {(r['stage'], int(r['seed']), r['role'], r['program']) for r in rows}
    programs = ('wphid', 'adam', 'gaussian_quasi_likelihood')
    roles = ('shared', 'partial', 'independent')
    expected = {('development', seed, role, program)
                for seed in range(9101101, 9101107) for role in roles for program in programs}
    expected.update({('fresh_conditional_comparison', seed, role, program)
                     for seed in range(9101301, 9101319) for role in roles for program in programs})
    if len(rows) != 216 or identities != expected:
        raise ValueError('Final216 comparative records missing or duplicated')
    summaries = {}
    for role in roles:
        summaries[role] = {}
        for program in programs:
            selected = [r for r in rows if r['stage'] == 'fresh_conditional_comparison'
                        and r['role'] == role and r['program'] == program]
            scores = [float(r['case_score']) for r in selected if r['case_score']]
            times = [float(r['cpu_seconds']) for r in selected if r['cpu_seconds']]
            summaries[role][program] = {
                'denominator': len(selected),
                'assessable': sum(r['assessable'] == 'True' for r in selected),
                'reaction_parameters_and_amplitudes_precision_passes': sum(r['precision_pass'] == 'True' for r in selected),
                'median_case_score_among_assessable': median(scores) if scores else None,
                'median_solver_CPU_seconds_among_recorded': median(times) if times else None,
                'budget_exhausted': sum(r['status'] == 'budget_exhausted' for r in selected),
            }
    c.save(OUTPUT / 'three-program-role-summary.json', {
        'fresh_noise_blocks': 18, 'public_observation_packets': 54,
        'given_family_active_support_and_partition': True,
        'precision_scope': 'Same-fit nonlinear reaction parameters and active amplitudes; base coefficients not gated',
        'descriptive_summary_only_no_new_hypothesis_tests': True,
        'source_CSV_sha256': c.sha(OUTPUT / 'comparison-cases.csv'),
        'roles': summaries,
    })
    required = [PAUSE, FRESH / 'recovery-source-freeze.json', FRESH / 'administrative-resume-context.json',
                FRESH / 'all-stages-complete.json', OUTPUT / 'recovered-completion-assessment.json',
                OUTPUT / 'plot-receipt.json', OUTPUT / 'three-program-role-summary.json']
    c.save(OUTPUT / 'orderly-resume-completion.json', {
        'version': 'COMPARISON_COMPLETION_ORDERLY_RESUME_V4_20261014',
        'bound_evidence_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in required},
        'resume_context': context,
        'study_unique_new_endpoint_fits': 198,
        'study_archived_WPHID_controls_reused': 18,
        'study_scored_records': 216,
        'this_resume_new_Gaussian_fits': 34,
        'this_resume_exact_completed_byte_reuses': 20,
        'orderly_pause_lost_model_or_CPU': 0,
        'postprocessing_new_fits_noise_or_integrations': 0,
        'source_sha256': c.sha(Path(__file__)),
        'previous_postprocessing_source_sha256': c.sha(ROOT / 'reproducibility/comparison_resume_v2/consolidate.py'),
        'full_open_estimator_adopted': False,
        'official_WENDy_scientific_runs': 0,
        'all_predeclared_locally_executable_scientific_comparisons_complete': True,
    })
    print(json.dumps({'all_declared_local_comparisons_complete': True, 'final_evidence': str(OUTPUT)}))


if __name__ == '__main__':
    main()
