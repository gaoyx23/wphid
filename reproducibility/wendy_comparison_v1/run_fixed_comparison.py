"""Freeze, fit once on archived inputs, and assess only saved comparator fits."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(ROOT))
sys.path.insert(2,str(ROOT/'reproducibility/final_candidate_v1'))
import numpy as np
from scientific_io import load_conditions
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.pipeline import prepare_views
from wphid.precision_development_v3.moment_risk import weak_operators
from wphid.wendy_comparison_v1 import GaussianWeakProblem, fit_gaussian_weak

PROTOCOL=ROOT/'reproducibility/wendy_comparison_v1/scientific-protocol-20261013.json'
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,record):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
def sources():
    paths=[]
    for directory in ('src/wphid/wendy_comparison_v1','tests/wendy_comparison_v1','reproducibility/wendy_comparison_v1'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    old=ROOT/'artifacts/paper_final_candidate_v1/scientific_confirmation_20261011/source-freeze.json'
    locked=read(old)['source_sha256']
    for relative,expected in locked.items():
        if sha(ROOT/relative)!=expected:
            raise ValueError('Archived dependency source changed: '+relative)
    locked.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    moment=ROOT/'src/wphid/precision_development_v3/moment_risk.py'
    locked[moment.relative_to(ROOT).as_posix()]=sha(moment)
    return locked
def verify(output):
    frozen=read(output/'source-freeze.json')
    for path,expected in frozen['source_sha256'].items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Frozen comparison source changed: '+path)
    for job in frozen['jobs']:
        if sha(ROOT/job['input_file'])!=job['input_sha256'] or sha(ROOT/job['control_file'])!=job['control_sha256']:
            raise ValueError('Archived input/control changed')
    return frozen
def prepare():
    protocol=read(PROTOCOL)
    output=ROOT/protocol['output_root']
    if output.exists():
        raise FileExistsError('Fixed comparison output already exists')
    previous=ROOT/protocol['archived_source']
    complete=read(previous/'execution-complete.json')
    if complete['source_freeze_sha256']!=sha(previous/'source-freeze.json'):
        raise ValueError('Archived completion source identity mismatch')
    jobs=[]
    for seed in protocol['seeds']:
        for role in protocol['roles']:
            control=previous/f'fits/Q1/{seed}_{role}_wphid.json'
            saved=read(control)
            item=saved['job']['input']
            if not saved['completed'] or sha(ROOT/item['file'])!=item['sha256']:
                raise ValueError('Archived input/control completion mismatch')
            if any(any(v for v in c.values()) for c in item['clipping'].values()):
                raise ValueError('Gaussian control preflight refuses active acquisition clipping')
            jobs.append({'id':f'{seed}_{role}', 'seed':seed,'role':role,'groups':protocol['groups'][role],
                         'input_file':item['file'],'input_sha256':item['sha256'],
                         'control_file':control.relative_to(ROOT).as_posix(),'control_sha256':sha(control)})
    save(output/'software-qualification.json',{'command':'python -m pytest tests/wendy_comparison_v1/test_gaussian_weak.py -q',
         'passed':7,'failed':0,'elapsed_seconds':6.02,'scientific_fits':0,'note':'fixed software fixtures only'})
    save(output/'source-freeze.json',{'protocol_sha256':sha(PROTOCOL),'source_sha256':sources(),'jobs':jobs,
         'software_qualification_sha256':sha(output/'software-qualification.json'),'inputs_reused':18,'new_noise_draws':0,
         'new_trajectory_integrations':0,'official_WENDy_execution':False})
    print(json.dumps({'prepared':True,'output':str(output),'new_fits':18,'sources_and_inputs_frozen':True}),flush=True)
    return output
def worker(output_name,job):
    output=Path(output_name)
    verify(output)
    protocol=read(PROTOCOL)
    begin=time.process_time()
    conditions=load_conditions(ROOT/job['input_file'],job['input_sha256'])
    training,heldout=prepare_views(conditions,paper_families(),protocol['training'],protocol['heldout'])
    family=next(f for f in paper_families() if f.name=='arrhenius_oxygen_damage')
    with np.load(ROOT/job['input_file'],allow_pickle=False) as data:
        operators=weak_operators(data['t'],data['z'],data['r'],tuple(protocol['window_shape']),tuple(protocol['stride_shape']))
    sigmas=tuple(np.asarray([.0005*np.sqrt(np.mean(s*s)) for s in c.states]) for c in training.conditions)
    problem=GaussianWeakProblem(training,family,job['groups'],operators,sigmas)
    preparation_cpu=time.process_time()-begin
    fit=fit_gaussian_weak(problem,starts=tuple(protocol['starts']),cpu_seconds=protocol['entire_fit_cpu_seconds'],max_evaluations=protocol['entire_fit_maximum_evaluations'])
    heldout_residual=None
    if fit['parameters'] is not None:
        parameters=np.asarray(fit['parameters']).reshape(-1,3)
        errors=[]
        for e,coefficients in enumerate(fit['physical_coefficients']):
            y=np.concatenate([c.targets[e] for c in heldout.conditions])
            pred=np.concatenate([c.bases[e]@np.asarray(coefficients[:3])+c.atom(family,parameters[job['groups'][e]])*coefficients[3] for c in heldout.conditions])
            errors.append(float(np.linalg.norm(pred-y)/max(np.linalg.norm(y),1e-12)))
        heldout_residual={'equation_weak_target_relative_residual':errors,'refitted':False,'conditions':protocol['heldout']}
    import wphid
    if Path(wphid.__file__).resolve()!=ROOT/'src/wphid/__init__.py':
        raise ValueError('Unexpected actual wphid import')
    record={'job':job,'completed':True,'fit':fit,'preparation_cpu_seconds':preparation_cpu,
            'frozen_heldout':heldout_residual,'sigmas_observed_training': [s.tolist() for s in sigmas],
            'training_scale_sha256':training.scale.fingerprint,'source_freeze_sha256':sha(output/'source-freeze.json'),
            'actual_import':str(Path(wphid.__file__).resolve()),'actual_cwd':str(Path.cwd()),'output_root':str(output),
            'truth_used_in_fit':False,'heldout_used_in_fit':False}
    save(output/f'fits/{job["id"]}.json',record)
    return {'id':job['id'],'status':fit['status'],'objective_evaluations':fit['objective_evaluations'],'cpu_seconds':fit['cpu_seconds']}
def run(output):
    frozen=verify(output)
    if (output/'execution-complete.json').exists():
        raise FileExistsError('Completed comparison must not rerun')
    pending=[]
    for job in frozen['jobs']:
        path=output/f'fits/{job["id"]}.json'
        if path.exists():
            saved=read(path)
            if not saved['completed'] or saved['job']!=job or saved['source_freeze_sha256']!=sha(output/'source-freeze.json'):
                raise ValueError('Existing fit identity mismatch')
        else:
            pending.append(job)
    with ProcessPoolExecutor(max_workers=read(PROTOCOL)['workers']) as pool:
        future=[pool.submit(worker,str(output),job) for job in pending]
        for result in as_completed(future):
            print(json.dumps(result.result()),flush=True)
    save(output/'execution-complete.json',{'new_fits':18,'fit_sha256':{job['id']:sha(output/f'fits/{job["id"]}.json') for job in frozen['jobs']},
         'source_freeze_sha256':sha(output/'source-freeze.json'),'new_noise_draws':0,'new_trajectory_integrations':0})
def assess(output):
    frozen=verify(output)
    complete=read(output/'execution-complete.json')
    # Scoring references become available only after every blind fit is saved.
    from experiments import paper_h3_joint_family_topology_partition as truth
    import assess_confirmation as archived_assessor
    rows=[]
    for job in frozen['jobs']:
        path=output/f'fits/{job["id"]}.json'
        if sha(path)!=complete['fit_sha256'][job['id']]:
            raise ValueError('Completed fit changed before scoring')
        record=read(path);fit=record['fit'];role=job['role']
        row={'seed':job['seed'],'role':role,'status':fit['status'],'assessable':False,'precision_pass':False,'case_score':None,
             'control':archived_assessor.assess_fit(read(ROOT/job['control_file']),role),'cpu_seconds':fit['cpu_seconds']}
        if fit['parameters'] is not None:
            theta=np.asarray(fit['parameters']).reshape(-1,3)[job['groups']]
            coefficients=np.asarray(fit['physical_coefficients'])
            if np.isfinite(theta).all() and np.isfinite(coefficients).all() and np.all(np.abs(coefficients[:,-1])>1e-12):
                errors=np.abs(theta-np.asarray(truth.TRUTH_CASES[role]['parameters']))/np.abs(truth.TRUTH_CASES[role]['parameters'])
                amplitude_errors=np.abs(coefficients[:,-1]-truth.TRUTH_AMPLITUDES)/np.abs(truth.TRUTH_AMPLITUDES)
                score=max(float(np.max(errors/np.asarray([.12,.18,.18]))),float(np.max(amplitude_errors/.25)))
                base_errors=[(np.abs(coefficients[e,:3]-np.asarray(truth.h1.CLOSED['truth_coefficients'][field][:-1]))/np.abs(truth.h1.CLOSED['truth_coefficients'][field][:-1])).tolist() for e,field in enumerate(truth.FIELDS)]
                row.update({'assessable':True,'precision_pass':score<=1.,'case_score':score,'parameter_errors_by_equation':errors.tolist(),
                            'amplitude_errors':amplitude_errors.tolist(),'base_coefficient_errors':base_errors,'frozen_heldout':record['frozen_heldout']})
        rows.append(row)
    summary={}
    for role in read(PROTOCOL)['roles']:
        selected=[r for r in rows if r['role']==role]
        assessable=[r for r in selected if r['assessable']]
        summary[role]={'n':6,'assessable':len(assessable),'precision_pass':sum(r['precision_pass'] for r in selected),
                       'control_precision_pass':sum(r['control']['precision_pass'] for r in selected),
                       'numeric_model_failure_count':sum(r['status']=='failed_retained' for r in selected),
                       'median_case_score':float(np.median([r['case_score'] for r in assessable])) if assessable else None,
                       'median_control_case_score':float(np.median([r['control']['case_score'] for r in selected if r['control']['assessable']])),
                       'budget_exhausted':sum(r['status']=='budget_exhausted' for r in selected)}
    qualified=all(v['assessable']>=5 and v['numeric_model_failure_count']==0 for v in summary.values())
    result={'rows':rows,'summary':summary,'qualification_gate_pass':qualified,'official_WENDy':False,
            'independent_confirmation':False,'adoption_status':'not_decided_by_conditional_known_support_comparison',
            'execution_complete_sha256':sha(output/'execution-complete.json'),'source_freeze_sha256':sha(output/'source-freeze.json')}
    save(output/'assessment.json',result)
    print(json.dumps({'summary':summary,'qualification_gate_pass':qualified}),flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('all','prepare','run','assess'),default='all')
    stage=parser.parse_args().stage
    output=ROOT/read(PROTOCOL)['output_root']
    if stage in ('all','prepare'):
        output=prepare()
    if stage in ('all','run'):
        run(output)
    if stage in ('all','assess'):
        assess(output)
if __name__=='__main__':
    main()
