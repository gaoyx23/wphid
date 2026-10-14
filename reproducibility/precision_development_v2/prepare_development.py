"""Freeze 12 changed-path fits and reuse six unchanged controls; no new inputs."""

from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import sys

from development_io import ROOT,OUTPUT,PRIOR,PROTOCOL,PROTOCOL_PATH,sha,read,write_new


def main():
    if Path.cwd().resolve()!=ROOT:raise ValueError('Wrong worktree')
    if (OUTPUT/'source-freeze.json').exists() or (OUTPUT/'input-manifest.json').exists():
        raise FileExistsError('Development already frozen; verify/read it, never regenerate')
    decision=read(ROOT/PROTOCOL['authorization']['record'])
    if not decision['scientific_execution']['supersedes_earlier_no_new_experiment_instruction']:
        raise PermissionError('Missing standing authorization for necessary new experiments')
    if PROTOCOL['previous_saved_confirmation_worker_CPU_seconds']+PROTOCOL['previous_interrupted_unsaved_CPU_upper_seconds']+PROTOCOL['maximum_new_saved_worker_CPU_seconds_arithmetic']>3600*PROTOCOL['combined_worker_CPU_budget_hours']:
        raise ValueError('Development arithmetic exceeds the combined worker budget')
    old_freeze=read(PRIOR/'source-freeze.json')
    old_ids={**old_freeze['source_sha256'],**old_freeze['frozen_reference_sha256']}
    for name,expected in old_ids.items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen V1 reference changed: '+name)
    qualifications=sorted((ROOT/'artifacts/paper_precision_development_v2/software_qualification').glob('attempt-*/qualification.json'))
    qpath=next((p for p in reversed(qualifications) if read(p)['all_passed']),None)
    if qpath is None:raise ValueError('No passed component software qualification')
    qualification=read(qpath)
    for name,expected in qualification['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source differs from software qualification: '+name)
    completion=read(PRIOR/'execution-complete.json')
    assessment=read(PRIOR/'assessment.json')
    saved_CPU=sum(read(ROOT/f)['total_worker_job_cpu_seconds'] for f in assessment['fit_source_sha256'])
    if abs(saved_CPU-PROTOCOL['previous_saved_confirmation_worker_CPU_seconds'])>1e-8:
        raise ValueError('Prior recorded-worker CPU differs from declared envelope')
    original_inputs=read(PRIOR/'input-manifest.json')
    jobs=[];controls=[]
    for seed in PROTOCOL['blocks']:
        for role in PROTOCOL['roles']:
            item=next(r for r in original_inputs['records'] if r['stage']=='Q1' and r['seed']==seed and r['role']==role)
            old_path=PRIOR/'fits/Q1'/f'{seed}_{role}_wphid.json'
            old=read(old_path)
            if not old['completed'] or old['status']!='success' or old['input_sha256']!=item['sha256']:
                raise ValueError('Original comparator/input is not a completed bound fit')
            relative=old_path.relative_to(ROOT).as_posix()
            if sha(old_path)!=assessment['fit_source_sha256'][relative] or sha(ROOT/item['file'])!=item['sha256']:
                raise ValueError('Selected old input/comparator identity changed')
            item_job={'id':f'{seed}_{role}_component_v2','seed':seed,'role':role,'stage':'conditional_development',
                         'program':'component_profile_v2','input':item,'groups':PROTOCOL['known_partitions'][role],
                         'V1_fit':relative,'V1_fit_sha256':sha(old_path),
                         'training_scale_sha256':old['training_scale_sha256']}
            if role in PROTOCOL['new_fit_roles']:jobs.append(item_job)
            else:controls.append({**item_job,'estimate_source':'unchanged_V1_control_reused','new_solver_calls':0})
    if (len(jobs)!=12 or len({j['id'] for j in jobs})!=12 or len(controls)!=6
            or any(j['role']=='shared' for j in jobs)):
        raise ValueError('Wrong changed-path job or unchanged-control count')
    sources={**old_ids,**{f:sha(ROOT/f) for f in PROTOCOL['source_files']}}
    sources[qpath.relative_to(ROOT).as_posix()]=sha(qpath)
    freeze={'version':PROTOCOL['version'],'frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
            'source_sha256':sources,'versions':{v:importlib.metadata.version(v) for v in ['numpy','scipy','threadpoolctl']},
            'python_version':sys.version,'python_executable':str(Path(sys.executable).resolve()),
            'software_qualification':qpath.relative_to(ROOT).as_posix(),'software_tests_passed':qualification['tests'],
            'new_estimates_before_freeze':0,'main_directory_modified':False,'old_results_modified':False}
    write_new(OUTPUT/'source-freeze.json',freeze)
    meta=[PRIOR/'source-freeze.json',PRIOR/'input-manifest.json',PRIOR/'execution-complete.json',PRIOR/'assessment.json',
          ROOT/PROTOCOL['authorization']['record']]
    manifest={'version':PROTOCOL['version'],'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),
              'jobs':jobs,'unchanged_controls':controls,'prior_metadata_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in meta},
              'new_observations':0,'new_weak_adaptations':0,'new_estimates':12,'reused_original_fits':18,
              'reused_unchanged_shared_controls':6,
              'development_inputs_previously_assessed':True,'independent_confirmation':False}
    write_new(OUTPUT/'input-manifest.json',manifest)
    print({'new_fit_jobs':len(jobs),'new_observations':0,'software_tests_passed':qualification['tests'],
           'maximum_new_worker_CPU_hours':2.2,'reused_shared_controls':6,'independent_confirmation':False})


if __name__=='__main__':main()
