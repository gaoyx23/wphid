"""Frozen Gaussian experiment with administrative paths and durable monitoring."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(1,str(ROOT))
PROTOCOL=Path(__file__).with_name('protocol.json')
P=json.loads(PROTOCOL.read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('frozen_gaussian_fresh_v2',ROOT/P['original_wrapper'])
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
OUTPUT=ROOT/P['output'];PARENT=ROOT/P['parent'];DEVELOPMENT=ROOT/P['development']

def bind():
    f.PARENT=PARENT;f.DEVELOPMENT=DEVELOPMENT;f.OUTPUT=OUTPUT

def save(path,value):
    if path.exists():raise FileExistsError('Never overwrite frozen scientific or audit output')
    f.save(path,value)

def verify_declarations():
    for name,expected in P['declaration_sha256'].items():
        if f.sha(ROOT/name)!=expected:raise ValueError('Pre-scoring scientific declaration changed')

def verify():
    bind();verify_declarations()
    frozen=f.verify()
    if len(frozen['jobs'])!=54 or frozen['new_Gaussian_fits_at_freeze']!=0:
        raise ValueError('Exactly54 previously frozen tasks required')
    return frozen

def prepare():
    bind();verify_declarations()
    if OUTPUT.exists():raise FileExistsError('Independent new output already exists')
    decision=f.read(DEVELOPMENT/'independent-entry-decision.json')
    if not decision['entered_fresh_Gaussian_comparison']:raise ValueError('Fixed development gate failed')
    qualified=f.read(Path(__file__).with_name('qualification.json'))
    if not qualified['all_passed']:raise ValueError('New orchestration not qualified')
    for name,expected in qualified['source_sha256'].items():
        if f.sha(ROOT/name)!=expected:raise ValueError('Qualified source changed')
    source,public=f.verify_parent()
    devsource=f.read(DEVELOPMENT/'source-freeze.json')
    for name,expected in devsource['source_sha256'].items():
        if f.sha(ROOT/name)!=expected:raise ValueError('Development kernel source changed')
    recovery=f.read(DEVELOPMENT/'recovery-source-freeze.json')
    for section in ('source_sha256','original_receipt_sha256','copied_fit_sha256'):
        for name,expected in recovery[section].items():
            if f.sha(ROOT/name)!=expected:raise ValueError('Recovered development identity changed')
    jobs=f.declared_jobs(public['records'],f.read(f.PROTO))
    locked=dict(devsource['source_sha256'])
    for section in ('source_sha256','old_source_sha256'):locked.update(source[section])
    locked.update(P['declaration_sha256'])
    files=[p for p in (ROOT/'reproducibility/wendy_comparison_v2').glob('*') if p.is_file()]
    files.extend((ROOT/'tests/wendy_comparison_v2').glob('*.py'))
    files.extend(p for p in Path(__file__).parent.glob('*') if p.is_file() and p.name!='qualification.json')
    files.extend((ROOT/'tests/wendy_comparison_resume_v3').glob('*.py'))
    locked.update({p.relative_to(ROOT).as_posix():f.sha(p) for p in files})
    receipts=[DEVELOPMENT/n for n in ('source-freeze.json','recovery-source-freeze.json','execution-complete.json','assessment.json','independent-entry-decision.json')]
    receipts.extend(PARENT/n for n in ('source-freeze.json','fresh-inputs.json','development-assessment.json','recovery-v2-source-freeze.json'))
    receipts.append(Path(__file__).with_name('qualification.json'))
    save(OUTPUT/'source-freeze.json',{'source_sha256':locked,'receipt_sha256':{p.relative_to(ROOT).as_posix():f.sha(p) for p in receipts},
        'jobs':jobs,'parent_source_freeze_sha256':f.sha(PARENT/'source-freeze.json'),
        'declared_control_jobs':[{'id':f"{j['id']}_{program}",'program':program,'stage':'fresh','input_sha256':j['input_sha256']} for j in jobs for program in ('wphid','adam')],
        'new_Gaussian_fits_at_freeze':0,'additional_noise_draws':0,'additional_trajectory_integrations':0,
        'completed_control_byte_hashes_frozen_before_comparative_assessment':True,'control_parameters_are_estimator_inputs':False,
        'administrative_wrapper_version':P['version'],'scientific_kernel_or_gate_changes':False})
    verify();print(json.dumps({'prepared':True,'fits':54,'output':str(OUTPUT)}),flush=True)

def run_one(job):
    bind();verify()
    import psutil
    process=psutil.Process()
    save(OUTPUT/f"attempt-starts/{job['id']}.json",{'id':job['id'],'pid':os.getpid(),'process_create_time':process.create_time(),
        'process_cpu_at_start':time.process_time(),'utc_epoch_at_start':time.time(),'source_freeze_sha256':f.sha(OUTPUT/'source-freeze.json')})
    result=f.worker(job)
    save(OUTPUT/f"attempt-completion/{job['id']}.json",{'id':job['id'],'result':result,'process_cpu_at_end':time.process_time(),
        'utc_epoch_at_end':time.time(),'start_receipt_sha256':f.sha(OUTPUT/f"attempt-starts/{job['id']}.json")})
    return result

def cpu_snapshot(start,process):
    if process.pid!=start['pid'] or abs(process.create_time()-start['process_create_time'])>1e-4:
        raise ValueError('CPU checkpoint process identity differs')
    cpu=process.cpu_times()
    return {'job_id':start['id'],'pid':process.pid,'job_CPU_seconds_observed':max(0.,cpu.user+cpu.system-start['process_cpu_at_start']),
            'process_create_time':process.create_time(),'rss_bytes':process.memory_info().rss}

def checkpoint(jobs,journal):
    import psutil
    states=[]
    for job in jobs:
        path=OUTPUT/f"attempt-starts/{job['id']}.json"
        if not path.exists():continue
        start=f.read(path)
        try:states.append(cpu_snapshot(start,psutil.Process(start['pid'])))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    journal.write(json.dumps({'utc_epoch':time.time(),'states':states})+'\n');journal.flush();os.fsync(journal.fileno())

def execute():
    frozen=verify()
    if (OUTPUT/'execution-complete.json').exists() or (OUTPUT/'driver-cpu-checkpoints.jsonl').exists() or (OUTPUT/'attempt-starts').exists():
        raise FileExistsError('Fresh attempt already started; do not silently retry')
    started=time.perf_counter()
    with (OUTPUT/'driver-cpu-checkpoints.jsonl').open('x',encoding='utf-8') as journal:
        with ProcessPoolExecutor(max_workers=P['workers']) as pool:
            futures={pool.submit(run_one,j):j for j in frozen['jobs']};active=set(futures)
            while active:
                done,active=wait(active,timeout=P['checkpoint_seconds'],return_when=FIRST_COMPLETED)
                checkpoint([futures[x] for x in active],journal)
                for future in done:print(json.dumps(future.result()),flush=True)
    verify()
    save(OUTPUT/'execution-complete.json',{'source_freeze_sha256':f.sha(OUTPUT/'source-freeze.json'),
        'fit_sha256':{j['id']:f.sha(OUTPUT/f"fits/{j['id']}.json") for j in frozen['jobs']},'new_Gaussian_fits':54,
        'additional_noise_draws':0,'additional_trajectory_integrations':0,'driver_CPU_ledger_sha256':f.sha(OUTPUT/'driver-cpu-checkpoints.jsonl'),
        'wall_seconds':time.perf_counter()-started,'administrative_retries':0})

def replay():
    import numpy as np
    frozen=verify();completion=f.read(OUTPUT/'execution-complete.json')
    if (OUTPUT/'saved-fit-replay-audit.json').exists():raise FileExistsError('Saved54 point replay already complete')
    protocol=f.read(ROOT/'reproducibility/wendy_comparison_v1/scientific-protocol-20261013.json');rows=[]
    for job in frozen['jobs']:
        path=OUTPUT/f"fits/{job['id']}.json"
        if f.sha(path)!=completion['fit_sha256'][job['id']]:raise ValueError('Saved point changed')
        record=f.read(path);fit=record['fit']
        if fit['parameters'] is None:
            rows.append({'id':job['id'],'replayed':False,'passed':False,'saved_failure_retained':True});continue
        conditions=f.load_conditions(ROOT/job['input_file'],job['input_sha256'])
        training,_=f.prepare_views(conditions,f.paper_families(),protocol['training'],protocol['heldout'])
        family=next(x for x in f.paper_families() if x.name=='arrhenius_oxygen_damage')
        with np.load(ROOT/job['input_file'],allow_pickle=False) as data:
            operators=f.weak_operators(data['t'],data['z'],data['r'],tuple(protocol['window_shape']),tuple(protocol['stride_shape']))
        problem=f.GaussianWeakProblem(training,family,job['groups'],operators,record['sigmas_observed_training'])
        vector=np.r_[(np.asarray(fit['parameters'])-problem.lower)/problem.span,np.asarray(fit['scaled_coefficients']).ravel()]
        value,gradient=problem.value_gradient(vector)
        error=abs(value-fit['quasi_negative_log_likelihood'])/max(1.,abs(fit['quasi_negative_log_likelihood']))
        physical=float(np.max(np.abs(np.asarray(problem.physical_coefficients(vector))-fit['physical_coefficients'])/np.maximum(1.,np.abs(fit['physical_coefficients']))))
        rows.append({'id':job['id'],'replayed':True,'relative_objective_error':error,'relative_physical_coefficient_error':physical,
            'gradient_finite':bool(np.isfinite(gradient).all()),'passed':error<=1e-9 and physical<=1e-12 and bool(np.isfinite(gradient).all())})
    save(OUTPUT/'saved-fit-replay-audit.json',{'rows':rows,'passed':sum(r['passed'] for r in rows),'replayed':sum(r['replayed'] for r in rows),
        'source_freeze_sha256':f.sha(OUTPUT/'source-freeze.json'),'execution_complete_sha256':f.sha(OUTPUT/'execution-complete.json'),
        'audit_source_sha256':f.sha(Path(__file__)),'new_parameter_fits':0,'new_noise_draws':0,'new_trajectory_integrations':0,'score_based_point_selection':False})

def cost():
    frozen=verify();records=[f.read(OUTPUT/f"fits/{j['id']}.json") for j in frozen['jobs']]
    save(OUTPUT/'administrative-cost-audit.json',{'selected_case_denominator':54,'new_retained_fits':54,'exact_completed_reuses':0,
        'selected_saved_solver_cpu_seconds':sum(r['fit']['cpu_seconds'] for r in records),
        'preparation_cpu_seconds':sum(r['preparation_cpu_seconds'] for r in records),
        'maximum_selected_solver_cpu_seconds':max(r['fit']['cpu_seconds'] for r in records),
        'nominal_solver_cpu_cap_seconds':600,'between_call_overshoot_possible':True,'administrative_retries':0,
        'lost_state_CPU_seconds':0,'one_saved_scored_point_per_declared_case':True,'official_WENDy':False,
        'driver_CPU_ledger_sha256':f.sha(OUTPUT/'driver-cpu-checkpoints.jsonl')})

def finish():
    execute()
    while not (PARENT/'fresh-execution-complete.json').exists():time.sleep(10)
    if (OUTPUT/'assessment.json').exists():raise FileExistsError('Comparative assessment already exists')
    bind();f.assess();replay();cost();verify()
    save(OUTPUT/'all-stages-complete.json',{'execution_sha256':f.sha(OUTPUT/'execution-complete.json'),
        'assessment_sha256':f.sha(OUTPUT/'assessment.json'),'replay_sha256':f.sha(OUTPUT/'saved-fit-replay-audit.json'),
        'cost_sha256':f.sha(OUTPUT/'administrative-cost-audit.json'),'new_Gaussian_fits':54,'new_noise_draws':0,'official_WENDy':False})
    print(json.dumps({'all_stages_complete':True,'output':str(OUTPUT)}),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'run':finish,'verify':verify}[sys.argv[1]]()
