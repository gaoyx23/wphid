"""Resume six administrative Gaussian tasks without changing the estimator."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(ROOT/'reproducibility/wendy_comparison_v1'))
sys.path.insert(2,str(ROOT/'reproducibility/final_candidate_v1'))
sys.path.insert(3,str(ROOT))
from run_fixed_comparison import read,sha,save,verify as original_verify,worker as original_worker,assess as original_assess
PROTOCOL=Path(__file__).with_name('protocol.json')
P=read(PROTOCOL)
ORIGINAL=ROOT/P['original_output']
OUTPUT=ROOT/P['output_root']

def classify(jobs,completed,protocol):
    expected_reused={f'{seed}_{role}' for seed in protocol['completed_reuse_seed_blocks'] for role in protocol['roles']}
    expected_pending={f'{seed}_{role}' for seed in protocol['pending_seed_blocks'] for role in protocol['roles']}
    if set(completed)!=expected_reused or {j['id'] for j in jobs}!=expected_reused|expected_pending:
        raise ValueError('Administrative completed/pending identities differ from authorized recovery')
    if not set(protocol['administratively_interrupted_jobs'])<=expected_pending:
        raise ValueError('Interrupted tasks outside declared pending set')
    return [j for j in jobs if j['id'] in expected_pending]

def verify_recovery():
    original_verify(OUTPUT)
    record=read(OUTPUT/'recovery-source-freeze.json')
    for section in ('source_sha256','original_receipt_sha256','copied_fit_sha256'):
        for path,expected in record[section].items():
            if sha(ROOT/path)!=expected:
                raise ValueError('Frozen recovery identity changed: '+path)
    return record

def prepare():
    if OUTPUT.exists():
        raise FileExistsError('New recovery output exists; never overwrite')
    original=original_verify(ORIGINAL)
    completed={p.stem:p for p in (ORIGINAL/'fits').glob('*.json')}
    pending=classify(original['jobs'],completed,P)
    qualification=read(Path(__file__).with_name('qualification.json'))
    if not qualification['all_passed']:
        raise ValueError('New administrative recovery qualification failed')
    for path,expected in qualification['source_sha256'].items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Qualified recovery source changed')
    originals=[ORIGINAL/'source-freeze.json',ORIGINAL/'software-qualification.json',
               ORIGINAL/'software-qualification-stdout.txt',ORIGINAL/'external-runtime-preflight.json']
    for source in originals:
        destination=OUTPUT/source.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            raise FileExistsError('Exact metadata copy exists')
        shutil.copyfile(source,destination)
        if sha(source)!=sha(destination):
            raise ValueError('Exact metadata copy differs')
    copies={}
    for identity,path in completed.items():
        record=read(path)
        job=next(j for j in original['jobs'] if j['id']==identity)
        if (not record['completed'] or record['job']!=job or record['source_freeze_sha256']!=sha(ORIGINAL/'source-freeze.json')):
            raise ValueError('Original completed fit unbound')
        destination=OUTPUT/'fits'/path.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
        if sha(destination)!=sha(path):
            raise ValueError('Completed fit byte copy differs')
        copies[destination.relative_to(ROOT).as_posix()]=sha(destination)
    files=[p for p in Path(__file__).parent.glob('*') if p.is_file()]
    files.extend((ROOT/'tests/wendy_comparison_recovery_v1').glob('*.py'))
    save(OUTPUT/'recovery-source-freeze.json',{'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
         'original_receipt_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in originals+list(completed.values())},
         'copied_fit_sha256':copies,'pending_jobs':pending,'original_source_freeze_sha256':sha(ORIGINAL/'source-freeze.json'),
         'selected_case_denominator':18,'completed_reuses':12,'new_fits_at_freeze':0,
         'interrupted_tasks':P['administratively_interrupted_jobs'],'lost_work_nominal_cpu_allowance_seconds':[0,1200],
         'lost_work_actual_cpu_unrecorded':True,'candidate_total_CPU_600_claim':False,
         'no_model_or_truth_based_retry':True,'new_noise_draws':0,'new_trajectory_integrations':0})
    print(json.dumps({'prepared':True,'output':str(OUTPUT),'exact_completed_reuses':12,'new_recovery_tasks':6}),flush=True)

def run_one(job):
    verify_recovery()
    start={'id':job['id'],'pid':os.getpid(),'process_cpu_seconds_at_start':time.process_time(),
           'utc_epoch_at_start':time.time(),'recovery_source_freeze_sha256':sha(OUTPUT/'recovery-source-freeze.json'),
           'administrative_lost_state_recovery':job['id'] in P['administratively_interrupted_jobs']}
    save(OUTPUT/f'attempt-starts/{job["id"]}.json',start)
    result=original_worker(str(OUTPUT),job)
    save(OUTPUT/f'attempt-completion/{job["id"]}.json',{'result':result,'pid':os.getpid(),
         'process_cpu_seconds_at_end':time.process_time(),'utc_epoch_at_end':time.time(),
         'start_receipt_sha256':sha(OUTPUT/f'attempt-starts/{job["id"]}.json')})
    return result

def ledger(stop):
    import psutil
    process=psutil.Process()
    destination=OUTPUT/'external-cpu-ledger.jsonl'
    with destination.open('x',encoding='utf-8') as stream:
        while True:
            children=[]
            for child in process.children(recursive=True):
                try:
                    cpu=child.cpu_times()
                    children.append({'pid':child.pid,'cpu_seconds':cpu.user+cpu.system,'rss_bytes':child.memory_info().rss})
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    pass
            stream.write(json.dumps({'utc_epoch':time.time(),'children':children})+'\n');stream.flush()
            if stop.wait(P['cpu_ledger_interval_seconds']):
                break

def execute():
    frozen=verify_recovery()
    if (OUTPUT/'execution-complete.json').exists():
        raise FileExistsError('Recovered comparison already complete')
    pending=[]
    for job in frozen['pending_jobs']:
        path=OUTPUT/f'fits/{job["id"]}.json'
        if path.exists():
            record=read(path)
            if not record['completed'] or record['job']!=job or record['source_freeze_sha256']!=sha(OUTPUT/'source-freeze.json'):
                raise ValueError('Existing recovered fit identity differs')
        else:
            pending.append(job)
    stop=threading.Event();thread=threading.Thread(target=ledger,args=(stop,),daemon=True);thread.start()
    try:
        with ProcessPoolExecutor(max_workers=P['workers']) as executor:
            futures=[executor.submit(run_one,job) for job in pending]
            for future in as_completed(futures):
                print(json.dumps(future.result()),flush=True)
    finally:
        stop.set();thread.join()
    verify_recovery()
    original=read(OUTPUT/'source-freeze.json')
    hashes={j['id']:sha(OUTPUT/f'fits/{j["id"]}.json') for j in original['jobs']}
    save(OUTPUT/'execution-complete.json',{'new_fits':6,'completed_fit_reuses':12,'selected_case_denominator':18,
         'fit_sha256':hashes,'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),
         'recovery_source_freeze_sha256':sha(OUTPUT/'recovery-source-freeze.json'),
         'lost_work_nominal_cpu_allowance_interval_seconds':[0,1200],'lost_work_actual_cpu_unrecorded':True,
         'external_cpu_ledger_sha256':sha(OUTPUT/'external-cpu-ledger.jsonl'),
         'new_noise_draws':0,'new_trajectory_integrations':0})

def assess():
    verify_recovery()
    complete=read(OUTPUT/'execution-complete.json')
    if complete['new_fits']+complete['completed_fit_reuses']!=18:
        raise ValueError('All18selected cases not present')
    original_assess(OUTPUT)

def main():
    command=sys.argv[1]
    {'prepare':prepare,'run':execute,'assess':assess,'verify':verify_recovery}[command]()
if __name__=='__main__':
    main()
