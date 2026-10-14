"""Resume only34 never-started Gaussian tasks, preserving20 exact endpoints."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(1,str(ROOT))
from reproducibility.wendy_comparison_resume_v3 import complete as c
f=c.f
PROTO=Path(__file__).with_name('protocol.json');P=f.read(PROTO)
OLD=ROOT/P['old_output'];OUTPUT=ROOT/P['output'];PAUSE=ROOT/P['pause_receipt']

def bind():
    c.OUTPUT=OUTPUT;c.bind()

def save(path,value):
    bind();c.save(path,value)

def pending_jobs(all_jobs,completed,pause_jobs):
    ids={j['id'] for j in all_jobs}
    pending=[j for j in all_jobs if j['id'] not in completed]
    if len(all_jobs)!=54 or len(ids)!=54 or len(completed)!=20 or not set(completed)<=ids:
        raise ValueError('Exactly54 declared with20 unique complete endpoints required')
    if pending!=pause_jobs or len(pending)!=34:
        raise ValueError('Only the34 never-started paused tasks may run')
    return pending

def copy_exact(source,target,expected):
    if f.sha(source)!=expected:raise ValueError('Completed source bytes changed')
    if target.exists():raise FileExistsError('Never overwrite an existing endpoint')
    if not target.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('Copy escaped output root')
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    if f.sha(target)!=expected:raise ValueError('Exact byte reuse differs')

def verify():
    bind();c.verify()
    frozen=f.read(OUTPUT/'recovery-source-freeze.json')
    for section in ('source_sha256','receipt_and_old_fit_sha256','copied_fit_sha256'):
        for name,expected in frozen[section].items():
            if f.sha(ROOT/name)!=expected:raise ValueError('Frozen orderly resume identity changed: '+name)
    return frozen

def prepare():
    if OUTPUT.exists():raise FileExistsError('New resume output already exists')
    c.OUTPUT=OLD;c.bind();original=c.verify()
    if f.sha(OLD/'source-freeze.json')!=P['retained_source_freeze_sha256']:raise ValueError('Original frozen experiment differs')
    pause=f.read(PAUSE)
    if pause['Gaussian_fresh_completed']!=20 or pause['Gaussian_fresh_remaining']!=34:raise ValueError('Pause denominator differs')
    qualified=f.read(Path(__file__).with_name('qualification.json'))
    if not qualified['all_passed']:raise ValueError('New administrative orchestration not qualified')
    for name,expected in qualified['source_sha256'].items():
        if f.sha(ROOT/name)!=expected:raise ValueError('Qualified new administrative source changed')
    complete={Path(name).stem:expected for name,expected in pause['fit_sha256'].items()}
    pending=pending_jobs(original['jobs'],complete,pause['remaining_jobs'])
    if {p.stem for p in (OLD/'attempt-starts').glob('*.json')}!=set(complete):raise ValueError('Paused unstarted set has a start record')
    if {p.stem for p in (OLD/'attempt-completion').glob('*.json')}!=set(complete):raise ValueError('Paused complete attempt records missing')
    oldhash={PAUSE.relative_to(ROOT).as_posix():f.sha(PAUSE),
        (OLD/'source-freeze.json').relative_to(ROOT).as_posix():f.sha(OLD/'source-freeze.json')}
    OUTPUT.mkdir(parents=True)
    copy_exact(OLD/'source-freeze.json',OUTPUT/'source-freeze.json',P['retained_source_freeze_sha256'])
    copies={}
    for name,expected in pause['fit_sha256'].items():
        path=ROOT/name;record=f.read(path)
        job=next(j for j in original['jobs'] if j['id']==path.stem)
        if not record['completed'] or record['job']!=job or record['source_freeze_sha256']!=P['retained_source_freeze_sha256']:
            raise ValueError('Old selected endpoint identity differs')
        target=OUTPUT/'fits'/path.name;copy_exact(path,target,expected)
        oldhash[name]=expected;copies[target.relative_to(ROOT).as_posix()]=expected
        for category in ('attempt-starts','attempt-completion'):
            old=OLD/category/path.name;oldhash[old.relative_to(ROOT).as_posix()]=f.sha(old)
    files=[p for p in Path(__file__).parent.glob('*') if p.is_file() and p.name!='qualification.json']
    files.extend((ROOT/'tests/wendy_comparison_resume_v4').glob('*.py'))
    save(OUTPUT/'recovery-source-freeze.json',{'version':P['version'],
        'source_sha256':{p.relative_to(ROOT).as_posix():f.sha(p) for p in files},
        'receipt_and_old_fit_sha256':oldhash,'copied_fit_sha256':copies,'pending_jobs':pending,
        'qualification_sha256':f.sha(Path(__file__).with_name('qualification.json')),
        'source_freeze_sha256':P['retained_source_freeze_sha256'],'completed_byte_reuses':20,'new_fits_at_freeze':0,
        'new_never_started_jobs':34,'unique_endpoint_denominator':54,'lost_model_or_CPU_at_orderly_pause':0,
        'scientific_retry':False,'scientific_kernel_or_gates_changed':False,'new_noise_or_integrations':0})
    verify();print(json.dumps({'prepared':True,'reuses':20,'new_tasks':34,'output':str(OUTPUT)}),flush=True)

def run_one(job):
    bind();verify()
    return c.run_one(job)

def execute():
    frozen=verify()
    if (OUTPUT/'execution-complete.json').exists() or (OUTPUT/'driver-cpu-checkpoints.jsonl').exists() or (OUTPUT/'attempt-starts').exists():
        raise FileExistsError('Resume attempt already started; audit rather than retry')
    started=time.perf_counter()
    with (OUTPUT/'driver-cpu-checkpoints.jsonl').open('x',encoding='utf-8') as journal:
        with ProcessPoolExecutor(max_workers=P['workers']) as pool:
            futures={pool.submit(run_one,j):j for j in frozen['pending_jobs']};active=set(futures)
            while active:
                done,active=wait(active,timeout=P['checkpoint_seconds'],return_when=FIRST_COMPLETED)
                c.checkpoint([futures[x] for x in active],journal)
                for future in done:print(json.dumps(future.result()),flush=True)
    verify();all_jobs=f.read(OUTPUT/'source-freeze.json')['jobs']
    save(OUTPUT/'execution-complete.json',{'source_freeze_sha256':f.sha(OUTPUT/'source-freeze.json'),
        'recovery_source_freeze_sha256':f.sha(OUTPUT/'recovery-source-freeze.json'),
        'fit_sha256':{j['id']:f.sha(OUTPUT/f"fits/{j['id']}.json") for j in all_jobs},
        'new_Gaussian_fits':54,'new_Gaussian_fits_field_scope':'unique endpoints in the whole fresh confirmation study',
        'completed_byte_reuses':20,'new_retained_resume_fits':34,'additional_noise_draws':0,'additional_trajectory_integrations':0,
        'driver_CPU_ledger_sha256':f.sha(OUTPUT/'driver-cpu-checkpoints.jsonl'),'wall_seconds':time.perf_counter()-started,
        'administrative_retries':0,'lost_model_or_CPU_at_orderly_pause':0})

def finish():
    execute();verify()
    if not (c.PARENT/'fresh-execution-complete.json').exists():raise ValueError('All108 parent controls must already be complete')
    bind();f.assess();c.replay();c.cost();verify()
    frozen=f.read(OUTPUT/'recovery-source-freeze.json')
    resumed=[f.read(OUTPUT/f"fits/{j['id']}.json") for j in frozen['pending_jobs']]
    save(OUTPUT/'administrative-resume-context.json',{'version':P['version'],'selected_unique_endpoint_denominator':54,
        'completed_byte_reuses':20,'new_retained_resume_fits':34,'original_study_fresh_Gaussian_fits':54,
        'cost_receipt_new_retained_fits_field_scope':'entire original fresh confirmation study;20 reused here and34 newly completed here',
        'new_retained_resume_solver_cpu_seconds':sum(r['fit']['cpu_seconds'] for r in resumed),
        'new_retained_resume_preparation_cpu_seconds':sum(r['preparation_cpu_seconds'] for r in resumed),
        'lost_model_or_CPU_at_orderly_pause':0,'scientific_retry_or_tuning':False,'new_noise_or_integrations':0,
        'original20_endpoint_bytes_unchanged':True,'all54_scored_only_after_completion':True,
        'cost_sha256':f.sha(OUTPUT/'administrative-cost-audit.json'),'recovery_source_freeze_sha256':f.sha(OUTPUT/'recovery-source-freeze.json')})
    save(OUTPUT/'all-stages-complete.json',{'execution_sha256':f.sha(OUTPUT/'execution-complete.json'),
        'assessment_sha256':f.sha(OUTPUT/'assessment.json'),'replay_sha256':f.sha(OUTPUT/'saved-fit-replay-audit.json'),
        'cost_sha256':f.sha(OUTPUT/'administrative-cost-audit.json'),'administrative_context_sha256':f.sha(OUTPUT/'administrative-resume-context.json'),
        'new_Gaussian_fits':54,'completed_byte_reuses':20,'new_retained_resume_fits':34,'new_noise_draws':0,'official_WENDy':False})
    print(json.dumps({'all_stages_complete':True,'output':str(OUTPUT)}),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'run':finish,'verify':verify}[sys.argv[1]]()
