"""Dependent fresh Gaussian comparator, using parent public inputs exclusively."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(ROOT/'reproducibility/wendy_comparison_v1'))
sys.path.insert(2,str(ROOT/'reproducibility/final_candidate_v1'))
sys.path.insert(3,str(ROOT))
import numpy as np
from run_fixed_comparison import read,sha,save,load_conditions,prepare_views,paper_families,weak_operators
from wphid.wendy_comparison_v1 import GaussianWeakProblem,fit_gaussian_weak

PROTO=Path(__file__).with_name('conditional-independent-protocol-20261013.json')
PARENT=ROOT/'artifacts/paper_matched_completion_v1'
OUTPUT=ROOT/'artifacts/paper_wendy_comparison_v2/fresh_conditional_confirmation_20261013_attempt01'
DEVELOPMENT=ROOT/read(PROTO)['development_source']

def declared_jobs(packets,protocol):
    expected={(seed,role) for seed in protocol['fresh_blocks'] for role in protocol['roles']}
    actual=[(p['seed'],p['role']) for p in packets]
    if len(actual)!=54 or len(set(actual))!=54 or set(actual)!=expected:
        raise ValueError('Exactly declared54 fresh seed/role packets required')
    result=[]
    for packet in packets:
        if not packet.get('parameters_not_present_in_estimator_input',False):
            raise ValueError('Estimator packet has no truth-free qualification')
        if any(any(v for v in c.values()) for c in packet['clipping'].values()):
            raise ValueError('Gaussian acquisition clipping is not qualified')
        result.append({'id':f"{packet['seed']}_{packet['role']}",'seed':packet['seed'],'role':packet['role'],
                       'groups':protocol['groups'][packet['role']], 'input_file':packet['file'],'input_sha256':packet['sha256']})
    return result

def validate_control(record,job,program,parent_source_sha):
    expected=f"{job['id']}_{program}"
    if (not record.get('completed') or record.get('source_freeze_sha256')!=parent_source_sha or
        record.get('input_sha256')!=job['input_sha256'] or record.get('job',{}).get('id')!=expected or
        record['job'].get('role')!=job['role'] or record['job'].get('program')!=program or
        record['job'].get('stage')!='fresh' or record['job'].get('input',{}).get('sha256')!=job['input_sha256']):
        raise ValueError('Completed parent control identity mismatch')

def verify_parent():
    source=read(PARENT/'source-freeze.json')
    for section in ('source_sha256','old_source_sha256','old_result_and_input_sha256'):
        for path,expected in source[section].items():
            if sha(ROOT/path)!=expected:
                raise ValueError('Parent frozen source/input changed: '+path)
    public=read(PARENT/'fresh-inputs.json')
    if public['source_freeze_sha256']!=sha(PARENT/'source-freeze.json') or not public['all_inputs_frozen_before_fresh_fits']:
        raise ValueError('Parent public inputs not frozen')
    return source,public

def verify():
    frozen=read(OUTPUT/'source-freeze.json')
    for path,expected in frozen['source_sha256'].items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Frozen fresh Gaussian source changed: '+path)
    for path,expected in frozen['receipt_sha256'].items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Frozen prerequisite receipt changed: '+path)
    for job in frozen['jobs']:
        if sha(ROOT/job['input_file'])!=job['input_sha256']:
            raise ValueError('Fresh public observation changed')
    import wphid
    if Path.cwd().resolve()!=ROOT or Path(wphid.__file__).resolve()!=ROOT/'src/wphid/__init__.py':
        raise ValueError('Unexpected actual cwd/import')
    return frozen

def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Fresh Gaussian output exists; never overwrite')
    decision=read(DEVELOPMENT/'independent-entry-decision.json')
    if not decision['entered_fresh_Gaussian_comparison']:
        raise ValueError('Dependent Gaussian entry gate failed')
    source,public=verify_parent()
    jobs=declared_jobs(public['records'],read(PROTO))
    locked=dict(read(DEVELOPMENT/'source-freeze.json')['source_sha256'])
    for section in ('source_sha256','old_source_sha256'):
        locked.update(source[section])
    files=[p for p in Path(__file__).parent.glob('*') if p.is_file()]
    files.extend((ROOT/'tests/wendy_comparison_v2').glob('*.py'))
    locked.update({p.relative_to(ROOT).as_posix():sha(p) for p in files})
    receipts=[DEVELOPMENT/'source-freeze.json',DEVELOPMENT/'execution-complete.json',DEVELOPMENT/'assessment.json',
              DEVELOPMENT/'independent-entry-decision.json',PARENT/'source-freeze.json',PARENT/'fresh-inputs.json',PARENT/'development-assessment.json']
    save(OUTPUT/'source-freeze.json',{'source_sha256':locked,'receipt_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in receipts},
         'jobs':jobs,'parent_source_freeze_sha256':sha(PARENT/'source-freeze.json'),
         'declared_control_jobs':[{'id':f"{j['id']}_{program}",'program':program,'stage':'fresh','input_sha256':j['input_sha256']}
                                  for j in jobs for program in ('wphid','adam')],
         'new_Gaussian_fits_at_freeze':0,'additional_noise_draws':0,'additional_trajectory_integrations':0,
         'completed_control_byte_hashes_frozen_before_comparative_assessment':True,'control_parameters_are_estimator_inputs':False})
    print(json.dumps({'fresh_Gaussian_prepared':True,'new_fits':54,'public_packets_reused':54}),flush=True)

def worker(job):
    verify()
    begin=time.process_time()
    old=read(ROOT/'reproducibility/wendy_comparison_v1/scientific-protocol-20261013.json')
    protocol=read(PROTO)
    conditions=load_conditions(ROOT/job['input_file'],job['input_sha256'])
    training,heldout=prepare_views(conditions,paper_families(),old['training'],old['heldout'])
    family=next(f for f in paper_families() if f.name=='arrhenius_oxygen_damage')
    with np.load(ROOT/job['input_file'],allow_pickle=False) as data:
        operators=weak_operators(data['t'],data['z'],data['r'],tuple(old['window_shape']),tuple(old['stride_shape']))
    sigma=tuple(np.asarray([.0005*np.sqrt(np.mean(s*s)) for s in c.states]) for c in training.conditions)
    problem=GaussianWeakProblem(training,family,job['groups'],operators,sigma)
    preparation_cpu=time.process_time()-begin
    fitted=fit_gaussian_weak(problem,starts=tuple(protocol['starts']),cpu_seconds=protocol['entire_fit_cpu_seconds'],
                             max_evaluations=protocol['entire_fit_maximum_evaluations'])
    heldout_report=None
    if fitted['parameters'] is not None:
        parameters=np.asarray(fitted['parameters']).reshape(-1,3)
        errors=[]
        for e,c in enumerate(fitted['physical_coefficients']):
            y=np.concatenate([s.targets[e] for s in heldout.conditions])
            predicted=np.concatenate([s.bases[e]@np.asarray(c[:3])+s.atom(family,parameters[job['groups'][e]])*c[3] for s in heldout.conditions])
            errors.append(float(np.linalg.norm(y-predicted)/max(np.linalg.norm(y),1e-12)))
        heldout_report={'conditions':old['heldout'],'refitted':False,'equation_weak_target_relative_residual':errors}
    record={'job':job,'completed':True,'fit':fitted,'frozen_heldout':heldout_report,'sigmas_observed_training':[s.tolist() for s in sigma],
            'preparation_cpu_seconds':preparation_cpu,'training_scale_sha256':training.scale.fingerprint,
            'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),'control_parameters_read':False,
            'truth_used_in_fit':False,'heldout_used_in_fit':False,'actual_cwd':str(Path.cwd()),'actual_import':str(ROOT/'src/wphid/__init__.py')}
    save(OUTPUT/f'fits/{job["id"]}.json',record)
    return {'id':job['id'],'status':fitted['status'],'cpu_seconds':fitted['cpu_seconds']}

def execute():
    frozen=verify()
    if (OUTPUT/'execution-complete.json').exists():
        raise FileExistsError('Fresh Gaussian comparison already complete')
    pending=[]
    for job in frozen['jobs']:
        path=OUTPUT/f'fits/{job["id"]}.json'
        if path.exists():
            record=read(path)
            if not record['completed'] or record['job']!=job or record['source_freeze_sha256']!=sha(OUTPUT/'source-freeze.json'):
                raise ValueError('Existing fresh Gaussian point changed')
        else:
            pending.append(job)
    with ProcessPoolExecutor(max_workers=read(PROTO)['workers']) as pool:
        future=[pool.submit(worker,job) for job in pending]
        for result in as_completed(future):
            print(json.dumps(result.result()),flush=True)
    verify()
    save(OUTPUT/'execution-complete.json',{'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),
         'fit_sha256':{j['id']:sha(OUTPUT/f'fits/{j["id"]}.json') for j in frozen['jobs']},'new_Gaussian_fits':54,
         'additional_noise_draws':0,'additional_trajectory_integrations':0})

def Gaussian_assessment_record(record):
    fitted=record['fit']
    if fitted['parameters'] is None or fitted['status']=='failed_retained':
        return {'status':'failed','fit':None}
    parameters=np.asarray(fitted['parameters']).reshape(-1,3)
    return {'status':'success','fit':{'hypothesis':[{'family':'arrhenius_oxygen_damage','groups':record['job']['groups'],
            'internal_parameters_by_group':parameters.tolist()}],'physical_coefficients':fitted['physical_coefficients'],
            'optimizer_status':fitted['status'],'cpu_seconds':fitted['cpu_seconds']},'frozen_heldout':record['frozen_heldout']}

def assess():
    frozen=verify()
    own=read(OUTPUT/'execution-complete.json')
    parent=read(PARENT/'fresh-execution-complete.json')
    if parent['source_freeze_sha256']!=frozen['parent_source_freeze_sha256']:
        raise ValueError('Parent control source identity differs')
    control_hashes={}
    for job in frozen['jobs']:
        for program in ('wphid','adam'):
            path=PARENT/f'fits/fresh/{job["id"]}_{program}.json'
            validate_control(read(path),job,program,frozen['parent_source_freeze_sha256'])
            relative=path.relative_to(ROOT).as_posix()
            if sha(path)!=parent['fit_sha256'][relative]:
                raise ValueError('Completed parent control bytes changed')
            control_hashes[relative]=sha(path)
        if sha(OUTPUT/f'fits/{job["id"]}.json')!=own['fit_sha256'][job['id']]:
            raise ValueError('Completed Gaussian fit changed')
    save(OUTPUT/'control-fit-identities.json',{'fit_sha256':control_hashes,'parent_completion_sha256':sha(PARENT/'fresh-execution-complete.json'),
         'own_completion_sha256':sha(OUTPUT/'execution-complete.json'),'frozen_before_comparative_truth_assessment':True})
    # All three complete programs and exact control bytes are now frozen.
    from assess_confirmation import assess_fit
    from scipy.stats import binomtest
    rows=[]
    names={'wphid':'WPHID','adam':'corrected_conditional_ADAM'}
    for job in frozen['jobs']:
        for program in ('wphid','adam'):
            record=read(PARENT/f'fits/fresh/{job["id"]}_{program}.json')
            rows.append({'seed':job['seed'],'program':names[program],**assess_fit(record,job['role'])})
        record=read(OUTPUT/f'fits/{job["id"]}.json')
        rows.append({'seed':job['seed'],'program':'conditional_Gaussian_weak_quasi_likelihood',
                     **assess_fit(Gaussian_assessment_record(record),job['role'])})
    lookup={(r['seed'],r['role'],r['program']):r for r in rows}
    clarification=read(Path(__file__).with_name('independent-directional-family-clarification-r1-20261013.json'))
    blocks=[]
    for seed in read(PROTO)['fresh_blocks']:
        scores={}
        for program in clarification['programs']:
            members=[lookup[seed,role,program] for role in read(PROTO)['roles']]
            scores[program]=max(r['case_score'] for r in members) if all(r['assessable'] for r in members) else None
        blocks.append({'seed':seed,'scores':scores})
    directions=[]
    for a,b in clarification['pairwise_program_pairs']:
        for winner,loser in ((a,b),(b,a)):
            wins=sum(x['scores'][winner] is not None and x['scores'][loser] is not None and x['scores'][loser]>0 and x['scores'][winner]<=.9*x['scores'][loser] for x in blocks)
            rates={role:{program:sum(r['precision_pass'] for r in rows if r['role']==role and r['program']==program)
                          for program in (winner,loser)} for role in read(PROTO)['roles']}
            rates_pass=all(v[winner]>=v[loser] for v in rates.values())
            directions.append({'winner':winner,'loser':loser,'block_wins':wins,'denominator':18,
                  'one_sided_binomial_p':float(binomtest(wins,18,.5,alternative='greater').pvalue),
                  'required_wins':15,'Bonferroni_alpha':.05/6,'precision_counts':rates,'precision_rate_gate':rates_pass,'passed':wins>=15 and rates_pass})
    save(OUTPUT/'assessment.json',{'rows':rows,'blocks':blocks,'directional_family':directions,'hypotheses':6,
         'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),'control_identities_sha256':sha(OUTPUT/'control-fit-identities.json'),
         'official_WENDy':False,'main_open_search_adoption':False,'noise_units':18,'additional_trajectory_integrations':0})
    print(json.dumps({'directions':directions}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','assess'))
    command=parser.parse_args().stage
    {'prepare':prepare,'run':execute,'assess':assess}[command]()
if __name__=='__main__':
    main()
