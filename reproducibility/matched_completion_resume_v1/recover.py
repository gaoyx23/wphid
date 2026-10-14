"""Administrative recovery into a separate root, without editing old kernels."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(ROOT))
from reproducibility.matched_completion_v1 import workflow as original
import psutil

PROTOCOL_PATH=Path(__file__).with_name('protocol.json')
P=original.read(PROTOCOL_PATH)
OLD=ROOT/P['original_root']
OUTPUT=ROOT/P['output_root']

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path,value,*,exclusive=True):
    path=Path(path)
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Recovery output escaped declared independent root')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def copied_exact(source,destination,expected):
    source,destination=Path(source),Path(destination)
    if original.sha(source)!=expected:
        raise ValueError('Original byte identity changed')
    if destination.exists():
        raise FileExistsError('Refuse to overwrite copied scientific record')
    if not destination.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Copy destination escaped independent output')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    if original.sha(destination)!=expected:
        raise ValueError('Copied record does not preserve original bytes')

def classify(declared,completed):
    pending=[j for j in declared if j['id'] not in completed]
    inflight=set(P['potential_inflight_administrative_recoveries'])
    if not inflight.issubset({j['id'] for j in pending}):
        raise ValueError('Declared administrative interruption set differs')
    return {'completed_reuses':len(completed),'pending':len(pending),
            'potential_inflight':len(inflight),'never_started_conditional_inference':len(pending)-len(inflight),
            'lost_nominal_CPU_seconds_interval':[0,len(inflight)*P['per_new_retained_attempt_maximum_CPU_seconds']],
            'unknown_overshoot_not_falsely_exact':True}

def bind_original():
    original.OUTPUT=OUTPUT

def verify():
    bind_original()
    original.verify_freeze()
    freeze=original.read(OUTPUT/'recovery-source-freeze.json')
    original.verify_hashes(freeze['recovery_source_sha256'])
    original.verify_hashes(freeze['original_control_sha256'])
    original.verify_hashes(freeze['reused_fit_and_observation_sha256'])
    if psutil.__version__!=P['psutil_version']:
        raise ValueError('Checkpoint dependency differs')
    return freeze

def prepare():
    original.OUTPUT=OLD
    original.verify_freeze()
    if OUTPUT.exists():
        raise FileExistsError('Recovery destination already exists; audit/read, do not replace')
    q=original.read(ROOT/'artifacts/paper_matched_completion_resume_v1/software/qualification.json')
    if not q['all_passed']:
        raise ValueError('Recovery orchestration not qualified')
    original.verify_hashes(q['source_sha256'])
    interruption=original.read(ROOT/P['interruption_receipt'])
    saved=interruption['fresh_paired_confirmation']['saved_fit_sha256']
    observed=list((OLD/'fits/fresh').glob('*.json'))
    if len(observed)!=12 or len(saved)!=12:
        raise ValueError('Completed snapshot is not exactly original twelve')
    control={}
    for name in ('source-freeze.json','fresh-inputs.json','development-inputs.json','development-assessment.json','development-execution-complete.json','external-selection-declaration-receipt.json'):
        src=OLD/name
        control[src.relative_to(ROOT).as_posix()]=original.sha(src)
    declaration=ROOT/P['selection_declaration']
    if original.sha(declaration)!=P['selection_declaration_sha256']:
        raise ValueError('Original pre-data selection declaration changed')
    control[declaration.relative_to(ROOT).as_posix()]=original.sha(declaration)
    control[P['interruption_receipt']]=original.sha(ROOT/P['interruption_receipt'])
    rows=original.read(OLD/'fresh-inputs.json')['records']
    declared=original.jobs('fresh',rows)
    reused={}
    for path in observed:
        expected=saved[path.relative_to(ROOT).as_posix()]
        record=original.read(path)
        job=next(j for j in declared if j['id']==record['job']['id'])
        original.validate_record(record,job)
        if original.sha(path)!=expected:
            raise ValueError('Saved interruption model changed')
        reused[job['id']]={'old_file':path.relative_to(ROOT).as_posix(),'sha256':expected}
    budget=classify(declared,reused)
    if budget['pending']!=96 or budget['never_started_conditional_inference']!=92:
        raise ValueError('Recovery count differs from frozen plan')
    OUTPUT.mkdir(parents=True)
    for name in ('source-freeze.json','fresh-inputs.json','development-inputs.json','development-assessment.json','development-execution-complete.json','external-selection-declaration-receipt.json'):
        src=OLD/name
        copied_exact(src,OUTPUT/name,control[src.relative_to(ROOT).as_posix()])
    for jobid,receipt in reused.items():
        copied_exact(ROOT/receipt['old_file'],OUTPUT/'fits/fresh'/f'{jobid}.json',receipt['sha256'])
    files=[*sorted(Path(__file__).parent.glob('*.py')),PROTOCOL_PATH,*sorted((ROOT/'tests/matched_completion_resume_v1').glob('*.py'))]
    freeze={'version':P['version'],'recovery_source_sha256':{p.relative_to(ROOT).as_posix():original.sha(p) for p in files},
            'original_control_sha256':control,
            'reused_fit_and_observation_sha256':{**{v['old_file']:v['sha256'] for v in reused.values()},**{r['file']:r['sha256'] for r in rows}},
            'original_source_freeze_sha256':original.sha(OUTPUT/'source-freeze.json'),
            'qualification_sha256':original.sha(ROOT/'artifacts/paper_matched_completion_resume_v1/software/qualification.json'),
            'new_retained_fits_at_recovery_freeze':0,'new_noise_packets':0,'new_integrations':0,
            'recorded_utc':now(),'actual_cwd':str(Path.cwd()),'actual_wphid_import':str(ROOT/'src/wphid/__init__.py'),
            'runtime_output_override_only':True,'psutil_version':psutil.__version__}
    save(OUTPUT/'recovery-source-freeze.json',freeze)
    save(OUTPUT/'recovery-plan.json',{'version':P['version'],'reused':reused,'budget_audit':budget,
         'pending_jobs':[j for j in declared if j['id'] not in reused],
         'original_inflight_model_and_CPU_not_recoverable':True,
         'new_retained_attempt_cap_seconds':600,'previous_lost_CPU_is_additional_not_hidden':True,
         'recovery_source_freeze_sha256':original.sha(OUTPUT/'recovery-source-freeze.json')})
    verify()
    print(json.dumps({'prepared':True,'root':str(OUTPUT),'budget':budget}),flush=True)

def initialize():
    bind_original()
    original.initializer()
    verify()

def run_one(job):
    bind_original()
    if original.job_path(job).exists():
        raise FileExistsError('Do not submit any completed or failed scientific job')
    process=psutil.Process()
    times=process.cpu_times()
    administrative=job['id'] in P['potential_inflight_administrative_recoveries']
    start={'job':job,'pid':os.getpid(),'pid_create_time':process.create_time(),'utc':now(),
           'monotonic_seconds':time.monotonic(),'process_CPU_seconds_at_start':times.user+times.system,
           'recovery_source_freeze_sha256':original.sha(OUTPUT/'recovery-source-freeze.json'),
           'input_sha256':job['input']['sha256'],'administrative_recovery_of_unavailable_old_point':administrative,
           'original_kernel_600_CPU_seconds_per_retained_attempt':True,'not_a_numerical_failure_retry':True}
    save(OUTPUT/'job-starts'/f"{job['id']}.json",start)
    result=original.run_job(job)
    save(OUTPUT/'job-finishes'/f"{job['id']}.json",{'result':result,'utc':now(),
         'fit_sha256':original.sha(original.job_path(job)),
         'job_start_sha256':original.sha(OUTPUT/'job-starts'/f"{job['id']}.json"),
         'recovery_source_freeze_sha256':original.sha(OUTPUT/'recovery-source-freeze.json')})
    return result

def checkpoint(jobs,journal):
    states=[]
    for job in jobs:
        startfile=OUTPUT/'job-starts'/f"{job['id']}.json"
        if not startfile.exists():continue
        try:
            start=original.read(startfile)
            process=psutil.Process(start['pid'])
            if process.create_time()!=start['pid_create_time']:continue
            cpu=process.cpu_times()
            total=cpu.user+cpu.system
            states.append({'job_id':job['id'],'pid':start['pid'],'process_CPU_seconds':total,
                           'job_CPU_seconds_observed':max(0.,total-start['process_CPU_seconds_at_start']),
                           'recorded_utc':now(),'monotonic_seconds':time.monotonic()})
        except (FileNotFoundError,json.JSONDecodeError,psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    journal.write(json.dumps({'recorded_utc':now(),'states':states},allow_nan=False)+'\n')
    journal.flush()
    os.fsync(journal.fileno())

def execute():
    verify()
    if (OUTPUT/'fresh-execution-complete.json').exists():
        raise FileExistsError('Complete confirmation never rerun')
    plan=original.read(OUTPUT/'recovery-plan.json')
    pending=plan['pending_jobs']
    if any(original.job_path(j).exists() for j in pending) or (OUTPUT/'driver-cpu-checkpoints.jsonl').exists():
        raise FileExistsError('A recovery attempt has already started; audit persisted budgets before another action')
    started=time.perf_counter()
    completed=[]
    with (OUTPUT/'driver-cpu-checkpoints.jsonl').open('x',encoding='utf-8') as journal:
        with ProcessPoolExecutor(max_workers=P['worker_processes'],initializer=initialize) as executor:
            futures={executor.submit(run_one,j):j for j in pending}
            active=set(futures)
            while active:
                done,active=wait(active,timeout=P['driver_checkpoint_interval_seconds'],return_when=FIRST_COMPLETED)
                checkpoint([futures[f] for f in active],journal)
                for future in done:
                    result=future.result()
                    completed.append(result)
                    save(OUTPUT/'recovery-status.json',{'retained_new_completed':len(completed),'new_pending_plan':96,
                         'original_completed_reused':12,'all_declared_completed':12+len(completed),
                         'last':result,'wall_seconds':time.perf_counter()-started,'recorded_utc':now()},exclusive=False)
                    print(json.dumps({'completed':12+len(completed),'total':108,'last':result}),flush=True)
    verify()
    declared=original.jobs('fresh',original.read(OUTPUT/'fresh-inputs.json')['records'])
    bound={}
    total=0.
    for job in declared:
        record=original.read(original.job_path(job))
        original.validate_record(record,job)
        bound[original.job_path(job).relative_to(ROOT).as_posix()]=original.sha(original.job_path(job))
        total+=record['total_worker_job_cpu_seconds']
    save(OUTPUT/'fresh-execution-complete.json',{'version':original.P['version'],'stage':'fresh','fit_sha256':bound,
         'source_freeze_sha256':original.sha(OUTPUT/'source-freeze.json'),'inputs_sha256':original.sha(OUTPUT/'fresh-inputs.json'),
         'new_fits':96,'completed_byte_reuses':12,'declared_scored_jobs':108,
         'completed_failures_retained':sum(r['status']=='failed' for r in completed),
         'worker_job_cpu_seconds':total,'new_retained_worker_job_cpu_seconds':sum(original.read(original.job_path(j))['total_worker_job_cpu_seconds'] for j in pending),
         'old_lost_CPU_nominal_allowance_interval_seconds':[0,2400],
         'old_unmeasured_between_call_overshoot_excluded_from_interval':True,
         'candidate_total_cap_unchanged_claim':False,'administrative_recovery_context':P['version'],
         'recovery_source_freeze_sha256':original.sha(OUTPUT/'recovery-source-freeze.json'),
         'CPU_checkpoint_journal_sha256':original.sha(OUTPUT/'driver-cpu-checkpoints.jsonl'),
         'wall_seconds':time.perf_counter()-started,'recorded_utc':now(),'truth_accuracy_assessed':False})
    print(json.dumps({'execution_complete':True,'declared':108,'reused':12,'new':96}),flush=True)

def assess():
    verify()
    original.assess('fresh')
    data=original.read(OUTPUT/'fresh-assessment.json')
    complete=original.read(OUTPUT/'fresh-execution-complete.json')
    save(OUTPUT/'administrative-assessment-context.json',{'fresh_assessment_sha256':original.sha(OUTPUT/'fresh-assessment.json'),
         'original_gates_unmodified':True,'one_scored_complete_attempt_per_job':True,
         'previous_partial_points_unavailable_not_selected':True,'old_lost_CPU_nominal_interval_seconds':[0,2400],
         'candidate_total_budget_unchanged_claim':False,'reused_completed_fits':12,'new_retained_attempts':96,
         'original_freeze_sha256':original.sha(OUTPUT/'source-freeze.json'),
         'recovery_freeze_sha256':original.sha(OUTPUT/'recovery-source-freeze.json'),
         'formal_conditional_advantage':data['formal_conditional_advantage'],
         'whole_method_adoption_from_this_comparison':False})

if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='run':execute()
    elif action=='assess':assess()
    elif action=='verify':verify();print('recovery freeze verified',flush=True)
    else:raise ValueError('Unknown recovery action')
