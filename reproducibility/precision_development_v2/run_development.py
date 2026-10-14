"""Checkpointed 12 changed-path fits; six shared controls and all V1 fits reused."""

from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from development_io import ROOT,OUTPUT,PROTOCOL,read,sha,write_new,verify_freeze
sys.path.insert(1,str(ROOT/'reproducibility/final_candidate_v1'))
from scientific_io import load_conditions
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import Candidate,Mechanism,FitBudget
from wphid.final_candidate_v1.pipeline import prepare_views,fit_record,frozen_residual_report
from wphid.precision_development_v2 import VERSION,fit_factored_candidate


def initialize_worker():
    global thread_limit
    from threadpoolctl import threadpool_limits,threadpool_info
    thread_limit=threadpool_limits(limits=1)
    verify_freeze()
    if any(p['num_threads']!=1 for p in threadpool_info()):
        raise ValueError('Numerical worker is not single-threaded')


def output_path(job):return OUTPUT/'fits'/(job['id']+'.json')


def verify_saved_record(record,job):
    if (record['job']!=job or not record['completed']
            or record['input_sha256']!=job['input']['sha256']
            or record['source_freeze_sha256']!=sha(OUTPUT/'source-freeze.json')
            or record['input_manifest_sha256']!=sha(OUTPUT/'input-manifest.json')):
        raise ValueError('Existing development estimate is not bound to this fixed job')


def run_job(job):
    if job not in read(OUTPUT/'input-manifest.json')['jobs']:
        raise ValueError('Job is outside the frozen development manifest')
    path=output_path(job)
    pending=list(path.parent.glob(path.name+'.pending-*'))
    if path.exists():
        previous=read(path);verify_saved_record(previous,job)
        return {'id':job['id'],'reused':True,'status':previous['status']}
    if pending:
        if len(pending)!=1:raise ValueError('Ambiguous interrupted saved record; no refit permitted')
        previous=read(pending[0]);verify_saved_record(previous,job)
        if not pending[0].resolve().is_relative_to(ROOT/'artifacts/paper_precision_development_v2'):
            raise ValueError('Interrupted record escaped V2 tree')
        pending[0].rename(path)
        return {'id':job['id'],'reused':True,'status':previous['status'],'saved_pending_promoted':True}
    started_cpu,started_wall=time.process_time(),time.perf_counter()
    conditions=load_conditions(ROOT/job['input']['file'],job['input']['sha256'])
    family=next(f for f in paper_families() if f.name==PROTOCOL['known_family'])
    training,heldout=prepare_views(conditions,(family,),tuple(PROTOCOL['train_conditions']),tuple(PROTOCOL['heldout_conditions']))
    if training.scale.fingerprint!=job['training_scale_sha256']:
        raise ValueError('Development train scale differs from original input-matched scale')
    candidate=Candidate((Mechanism('reaction',family,tuple(job['groups'])),))
    setup=time.process_time()-started_cpu
    record={'job':job,'completed':True,'input_sha256':job['input']['sha256'],
            'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),
            'input_manifest_sha256':sha(OUTPUT/'input-manifest.json'),
            'training_scale_sha256':training.scale.fingerprint,
            'actual_import':str(Path(sys.modules['wphid'].__file__).resolve()),
            'solver_version':VERSION,'threads':1,'device':'CPU','dtype':'float64',
            'truth_used_in_fitting':False,'heldout_used_in_fitting':False,'V1_point_warm_start':False,
            'independent_confirmation':False,'new_observations':0,
            'scientific_result_identity':'new algorithm-development estimate on reused observations',
            'shared_input_loading_and_scaling_cpu_seconds':setup}
    try:
        allowance=min(PROTOCOL['fit_budget']['maximum_cpu_seconds'],
                      PROTOCOL['worker_job_CPU_guard_seconds']-setup-PROTOCOL['post_fit_CPU_reserve_seconds'])
        if allowance<=0:raise RuntimeError('Worker setup exhausted the declared program allowance')
        settings=dict(PROTOCOL['fit_budget']);settings['maximum_cpu_seconds']=allowance
        settings['seeds']=tuple(settings['seeds']);settings['initial_fractions']=tuple(settings['initial_fractions'])
        budget=FitBudget(**settings)
        result=fit_factored_candidate(training,candidate,budget)
        record.update({'status':'success','budget':asdict(budget),'fit':fit_record(training,result.fit),
                       'flat_parameters':result.fit.parameters.tolist(),'component_layout':asdict(result.factorization),
                       'component_solver_audit':result.audit,'frozen_heldout':frozen_residual_report(heldout,result.fit)})
    except (RuntimeError,FloatingPointError,ValueError) as error:
        record.update({'status':'failed','error_type':type(error).__name__,'error':str(error),
                       'failure_retained_in_denominator':True})
    record['total_worker_job_cpu_seconds']=time.process_time()-started_cpu
    record['total_worker_job_wall_seconds']=time.perf_counter()-started_wall
    record['worker_CPU_guard_exceeded']=record['total_worker_job_cpu_seconds']>PROTOCOL['worker_job_CPU_guard_seconds']
    write_new(path,record)
    return {'id':job['id'],'reused':False,'status':record['status'],
            'CPU_seconds':record['total_worker_job_cpu_seconds'],
            'optimizer_status':record.get('fit',{}).get('optimizer_status')}


def main():
    manifest=verify_freeze()
    if (OUTPUT/'execution-complete.json').exists():
        raise FileExistsError('Development complete; read it, never rerun')
    started=OUTPUT/'execution-started.json'
    if not started.exists():
        write_new(started,{'started_at_UTC':datetime.now(timezone.utc).isoformat(),
                          'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),
                          'input_manifest_sha256':sha(OUTPUT/'input-manifest.json'),'declared_jobs':PROTOCOL['new_estimates']})
    finished=[]
    with ProcessPoolExecutor(max_workers=PROTOCOL['concurrency']['workers'],initializer=initialize_worker) as pool:
        futures=[pool.submit(run_job,job) for job in manifest['jobs']]
        for future in as_completed(futures):
            result=future.result();finished.append(result)
            print({'completed':len(finished),'total':len(futures),**result},flush=True)
    records=[]
    for job in manifest['jobs']:
        record=read(output_path(job));verify_saved_record(record,job);records.append(record)
    verify_freeze()
    write_new(OUTPUT/'execution-complete.json',{
        'completed_at_UTC':datetime.now(timezone.utc).isoformat(),
        'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),
        'input_manifest_sha256':sha(OUTPUT/'input-manifest.json'),
        'completed_jobs':len(records),'failed_jobs':sum(r['status']!='success' for r in records),
        'new_estimates_this_invocation':sum(not r['reused'] for r in finished),
        'saved_worker_CPU_seconds':sum(r['total_worker_job_cpu_seconds'] for r in records),
        'worker_guard_violations':[r['job']['id'] for r in records if r['worker_CPU_guard_exceeded']],
        'new_fit_source_sha256':{output_path(j).relative_to(ROOT).as_posix():sha(output_path(j)) for j in manifest['jobs']},
        'reused_V1_fits':18,'reused_unchanged_shared_controls':6,
        'new_observations':0,'independent_confirmation':False})


if __name__=='__main__':main()
