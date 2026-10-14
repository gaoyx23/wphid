"""Finish only three lost-state administrative attempts, preserving 105 points."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(1,str(ROOT))
from reproducibility.matched_completion_resume_v1 import recover as r1
w=r1.original
PROTOCOL_PATH=Path(__file__).with_name('protocol.json')
P=w.read(PROTOCOL_PATH)
OLD=ROOT/P['old_root'];OUTPUT=ROOT/P['output_root']

def bind():
    r1.OUTPUT=OUTPUT
    r1.P={**r1.P,'potential_inflight_administrative_recoveries':P['pending_ids']}
    r1.bind_original()

def save(path,value,**kwargs):
    bind();r1.save(path,value,**kwargs)

def copy_exact(source,target,expected):
    if w.sha(source)!=expected:raise ValueError('Old point byte identity changed')
    if target.exists():raise FileExistsError('Never overwrite an old/recovery point')
    if not target.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('Copy root escaped')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    if w.sha(target)!=expected:raise ValueError('Copied bytes differ')

def lost_audit(last):
    states={s['job_id']:s for s in last['states']}
    if set(states)!=set(P['pending_ids']):raise ValueError('Lost state identities differ')
    lower=sum(states[j]['job_CPU_seconds_observed'] for j in P['pending_ids'])
    if abs(lower-1189.625)>1e-6:raise ValueError('Lost CPU lower bound differs')
    return {'observed_lower_CPU_seconds':lower,'nominal_allowance_upper_CPU_seconds':1800,
            'last_checkpoint_utc':last['recorded_utc'],'unobserved_gap_and_overshoot_separate':True,
            'exact_lost_CPU_or_model_state_available':False,'not_an_algorithm_failure':True}

def verify():
    bind();r1.verify()
    f=w.read(OUTPUT/'recovery-v2-source-freeze.json')
    w.verify_hashes(f['source_sha256']);w.verify_hashes(f['old_control_and_fit_sha256'])
    return f

def prepare():
    r1.OUTPUT=OLD;r1.bind_original();r1.verify()
    if OUTPUT.exists():raise FileExistsError('V2 output already exists; never replace')
    q=w.read(ROOT/'artifacts/paper_matched_completion_resume_v2/software/qualification.json')
    if not q['all_passed']:raise ValueError('V2 new software not qualified')
    w.verify_hashes(q['source_sha256'])
    declared=w.jobs('fresh',w.read(OLD/'fresh-inputs.json')['records'])
    fits=list((OLD/'fits/fresh').glob('*.json'))
    if len(fits)!=105:raise ValueError('Exactly105 saved points required')
    complete_ids=set();oldhash={}
    for p in fits:
        record=w.read(p);job=next(j for j in declared if j['id']==record['job']['id'])
        w.validate_record(record,job)
        complete_ids.add(job['id']);oldhash[p.relative_to(ROOT).as_posix()]=w.sha(p)
    pending=[j for j in declared if j['id'] not in complete_ids]
    if {j['id'] for j in pending}!=set(P['pending_ids']):raise ValueError('Only lastthree may be run')
    journal=OLD/'driver-cpu-checkpoints.jsonl'
    last=json.loads(journal.read_text(encoding='utf-8').splitlines()[-1]);lost=lost_audit(last)
    names=('source-freeze.json','fresh-inputs.json','development-inputs.json','development-assessment.json',
           'development-execution-complete.json','external-selection-declaration-receipt.json','recovery-source-freeze.json')
    for name in names:oldhash[(OLD/name).relative_to(ROOT).as_posix()]=w.sha(OLD/name)
    oldhash[journal.relative_to(ROOT).as_posix()]=w.sha(journal)
    oldhash[(OLD/'recovery-plan.json').relative_to(ROOT).as_posix()]=w.sha(OLD/'recovery-plan.json')
    OUTPUT.mkdir(parents=True)
    for name in names:copy_exact(OLD/name,OUTPUT/name,oldhash[(OLD/name).relative_to(ROOT).as_posix()])
    for path in fits:copy_exact(path,OUTPUT/'fits/fresh'/path.name,oldhash[path.relative_to(ROOT).as_posix()])
    files=[*sorted(Path(__file__).parent.glob('*.py')),PROTOCOL_PATH,*sorted((ROOT/'tests/matched_completion_resume_v2').glob('*.py'))]
    save(OUTPUT/'recovery-v2-source-freeze.json',{'version':P['version'],'source_sha256':{p.relative_to(ROOT).as_posix():w.sha(p) for p in files},
         'old_control_and_fit_sha256':oldhash,'recovery_v1_freeze_sha256':w.sha(OUTPUT/'recovery-source-freeze.json'),
         'qualification_sha256':w.sha(ROOT/'artifacts/paper_matched_completion_resume_v2/software/qualification.json'),
         'frozen_before_three_retained_attempts':True,'new_fits_at_freeze':0,'new_inputs':0,'recorded_utc':r1.now()})
    save(OUTPUT/'recovery-v2-plan.json',{'pending_jobs':pending,'completed_byte_reuses':105,'lost_CPU':lost,
         'first_interruption_nominal_lost_interval_seconds':[0,2400],'retained_attempt_CPU_cap_seconds':600,
         'candidate_total_600_budget_unchanged_claim':False,'old_lost_points_not_selected':True,
         'source_freeze_sha256':w.sha(OUTPUT/'recovery-v2-source-freeze.json')})
    verify();print(json.dumps({'prepared':True,'reused':105,'new':3,'root':str(OUTPUT),'lost':lost}),flush=True)

def initializer():
    bind();r1.initialize();verify()

def run_one(job):
    bind();return r1.run_one(job)

def replay():
    import numpy as np
    from scientific_io import load_conditions
    from assess_confirmation import restore_fit
    from wphid.final_candidate_v1.problem import paper_families
    from wphid.final_candidate_v1.pipeline import prepare_views,frozen_residual_report
    from wphid.final_candidate_v1.solver import predict_frozen,physical_coefficients
    rows=[];bound={}
    jobs=w.jobs('fresh',w.read(OUTPUT/'fresh-inputs.json')['records'])
    for job in jobs:
        path=w.job_path(job);record=w.read(path);w.validate_record(record,job);bound[path.relative_to(ROOT).as_posix()]=w.sha(path)
        if record['status']!='success':
            rows.append({'id':job['id'],'saved_failure_retained':True,'replayed':False});continue
        conditions=load_conditions(ROOT/job['input']['file'],job['input']['sha256'])
        family=next(f for f in paper_families() if f.name==w.ORIGINAL['Q1']['known_family'])
        training,heldout=prepare_views(conditions,(family,),tuple(w.ORIGINAL['train_conditions']),tuple(w.ORIGINAL['heldout_conditions']))
        fit=restore_fit(record)
        if training.scale.fingerprint!=record['training_scale_sha256']:raise ValueError('Replay scale differs')
        predicted=predict_frozen(training,fit)
        targets=tuple(np.concatenate([c.targets[e] for c in training.conditions]) for e in range(len(predicted)))
        rss=np.asarray([np.sum((y-p)**2) for y,p in zip(targets,predicted)])
        np.testing.assert_allclose(rss,record['fit']['equation_training_rss'],rtol=2e-8,atol=2e-10)
        for left,right in zip(physical_coefficients(training,fit),record['fit']['physical_coefficients']):
            np.testing.assert_allclose(left,right,rtol=3e-13,atol=1e-13)
        held=frozen_residual_report(heldout,fit)
        np.testing.assert_allclose(held['equation_weak_target_relative_residual'],record['frozen_heldout']['equation_weak_target_relative_residual'],rtol=3e-13,atol=1e-13)
        rows.append({'id':job['id'],'replayed':True,'maximum_absolute_RSS_difference':float(np.max(np.abs(rss-np.asarray(record['fit']['equation_training_rss'])))),
                     'heldout_refit':False,'new_parameter_fit':False})
    verify()
    save(OUTPUT/'saved-point-replay.json',{'version':P['version'],'rows':rows,'declared':108,'replayed_successes':sum(r.get('replayed',False) for r in rows),
        'retained_saved_failures':sum(r.get('saved_failure_retained',False) for r in rows),'fit_sha256':bound,
        'source_freeze_sha256':w.sha(OUTPUT/'recovery-v2-source-freeze.json'),'new_fits_or_noise':0,
        'scientific_precision_success_count':None,'original_sources_unchanged':True,'recorded_utc':r1.now()})

def execute():
    verify()
    if (OUTPUT/'fresh-execution-complete.json').exists():raise FileExistsError('Already complete; do not rerun')
    plan=w.read(OUTPUT/'recovery-v2-plan.json');pending=plan['pending_jobs']
    if (OUTPUT/'driver-cpu-checkpoints.jsonl').exists() or any(w.job_path(j).exists() for j in pending):
        raise FileExistsError('V2 attempt already started; audit rather than retry')
    started=time.perf_counter();done_records=[]
    with (OUTPUT/'driver-cpu-checkpoints.jsonl').open('x',encoding='utf-8') as journal:
        with ProcessPoolExecutor(max_workers=3,initializer=initializer) as executor:
            futures={executor.submit(run_one,j):j for j in pending};active=set(futures)
            while active:
                done,active=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                r1.checkpoint([futures[f] for f in active],journal)
                for future in done:
                    result=future.result();done_records.append(result)
                    save(OUTPUT/'recovery-v2-status.json',{'completed':105+len(done_records),'total':108,'last':result,'recorded_utc':r1.now()},exclusive=False)
                    print(json.dumps({'completed':105+len(done_records),'total':108,'last':result}),flush=True)
    verify()
    jobs=w.jobs('fresh',w.read(OUTPUT/'fresh-inputs.json')['records']);bound={};records=[]
    for job in jobs:
        record=w.read(w.job_path(job));w.validate_record(record,job);records.append(record)
        bound[w.job_path(job).relative_to(ROOT).as_posix()]=w.sha(w.job_path(job))
    save(OUTPUT/'fresh-execution-complete.json',{'version':w.P['version'],'stage':'fresh','fit_sha256':bound,
        'source_freeze_sha256':w.sha(OUTPUT/'source-freeze.json'),'inputs_sha256':w.sha(OUTPUT/'fresh-inputs.json'),
        'new_fits':3,'completed_byte_reuses':105,'declared_scored_jobs':108,'completed_failures_retained':sum(r['status']=='failed' for r in records),
        'worker_job_cpu_seconds':sum(r['total_worker_job_cpu_seconds'] for r in records),
        'new_retained_worker_job_cpu_seconds':sum(w.read(w.job_path(j))['total_worker_job_cpu_seconds'] for j in pending),
        'lost_CPU_second_interruption':plan['lost_CPU'],'lost_CPU_first_interruption_nominal_interval_seconds':[0,2400],
        'candidate_total_600_budget_unchanged_claim':False,'one_scored_complete_attempt_per_job':True,
        'recovery_v2_source_freeze_sha256':w.sha(OUTPUT/'recovery-v2-source-freeze.json'),
        'CPU_checkpoint_journal_sha256':w.sha(OUTPUT/'driver-cpu-checkpoints.jsonl'),
        'wall_seconds':time.perf_counter()-started,'recorded_utc':r1.now(),'truth_accuracy_assessed':False})
    w.assess('fresh')
    replay()
    verify()
    save(OUTPUT/'administrative-assessment-context.json',{'version':P['version'],'fresh_assessment_sha256':w.sha(OUTPUT/'fresh-assessment.json'),
        'replay_sha256':w.sha(OUTPUT/'saved-point-replay.json'),'source_freeze_sha256':w.sha(OUTPUT/'recovery-v2-source-freeze.json'),
        'old_completed_byte_reuses':105,'new_retained_attempts':3,'one_complete_scored_attempt_per_job':True,
        'old_lost_points_unavailable_and_not_selected':True,'old_105_points_unchanged':True,
        'original_gates_and_objective_kernels_unchanged':True,'candidate_total_600_budget_unchanged_claim':False,
        'lost_CPU_second_interruption':plan['lost_CPU'],'lost_CPU_first_interruption_nominal_interval_seconds':[0,2400],
        'pure_same_loss_optimizer_comparison':False,'new_noise_or_integrations':0,'whole_method_adoption':False,
        'recorded_utc':r1.now()})
    print(json.dumps({'execution_assessment_replay_complete':True,'root':str(OUTPUT)}),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':execute()
    elif sys.argv[1]=='verify':verify();print('V2 freeze verified',flush=True)
    else:raise ValueError('Unknown command')
