"""Fixed conditional ADAM correction and dependent fresh matched confirmation.

Original WPHID development estimates are read and hash-bound, never rerun.
Terminal ADAM estimates are not profiled/refined after fitting.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / 'reproducibility/final_candidate_v1'))
PROTOCOL_PATH = Path(__file__).with_name('protocol.json')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

P = read(PROTOCOL_PATH)
OUTPUT = ROOT / P['output_root']
OLD = ROOT / P['old_output']
ORIGINAL = read(ROOT / P['unchanged_original_protocol'])

def save(path, value, *, exclusive=True):
    path = Path(path)
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Output escaped independent matched completion root')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def verify_hashes(mapping):
    for name, expected in mapping.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen identity changed: ' + name)

def verify_runtime():
    import wphid
    if Path.cwd().resolve() != ROOT or Path(wphid.__file__).resolve() != ROOT / 'src/wphid/__init__.py':
        raise ValueError('Wrong cwd or actual wphid import')
    lock = read(ROOT / 'reproducibility/final_candidate_v1/runtime-lock.json')
    for name, version in lock['numerical_test_dependency_closure'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Runtime changed: ' + name)

def verify_freeze():
    verify_runtime()
    freeze = read(OUTPUT / 'source-freeze.json')
    verify_hashes(freeze['source_sha256'])
    verify_hashes(freeze['old_source_sha256'])
    verify_hashes(freeze['old_result_and_input_sha256'])
    return freeze

def old_receipts():
    oldfreeze = read(OLD / 'source-freeze.json')
    verify_hashes({**oldfreeze['source_sha256'], **oldfreeze['frozen_reference_sha256']})
    manifest = read(OLD / 'input-manifest.json')
    if manifest['source_freeze_sha256'] != sha(OLD / 'source-freeze.json'):
        raise ValueError('Old input manifest freeze differs')
    q = read(ROOT / P['known_support_qualification'])
    if not q['all_passed']:
        raise ValueError('Known-support adapter not qualified')
    verify_hashes(q['source_sha256'])
    verify_hashes(q['prior_source_sha256'])
    rows, bound = [], {}
    for packet in manifest['records']:
        if packet['stage'] != 'Q1' or packet['seed'] not in P['development_seeds']:
            continue
        path = OLD / 'fits/Q1' / f"{packet['seed']}_{packet['role']}_wphid.json"
        record = read(path)
        if (not record['completed'] or record['input_sha256'] != packet['sha256']
                or record['job']['input'] != packet
                or record['source_freeze_sha256'] != sha(OLD / 'source-freeze.json')):
            raise ValueError('Unbound old WPHID estimate')
        if sha(ROOT / packet['file']) != packet['sha256']:
            raise ValueError('Old observation differs')
        rows.append({'input':packet, 'old_wphid_file':path.relative_to(ROOT).as_posix(),
                     'old_wphid_sha256':sha(path)})
        bound[packet['file']] = packet['sha256']
        bound[path.relative_to(ROOT).as_posix()] = sha(path)
    if len(rows) != 18:
        raise ValueError('Exactly 18 development reuse pairs required')
    for path in (OLD / 'source-freeze.json', OLD / 'input-manifest.json', OLD / 'execution-complete.json',
                 ROOT / P['known_support_qualification']):
        bound[path.relative_to(ROOT).as_posix()] = sha(path)
    return rows, bound, {**oldfreeze['source_sha256'],**oldfreeze['frozen_reference_sha256'],**q['source_sha256'],**q['prior_source_sha256']}

def prepare():
    verify_runtime()
    if (OUTPUT / 'source-freeze.json').exists():
        raise FileExistsError('Already frozen; verify/resume, never prepare again')
    qualification = read(OUTPUT / 'software/qualification.json')
    if not qualification['all_passed']:
        raise ValueError('New orchestration qualification missing')
    verify_hashes(qualification['source_sha256'])
    rows, bound, oldsources = old_receipts()
    files = [*sorted(Path(__file__).parent.glob('*.py')), PROTOCOL_PATH,
             *sorted((ROOT / 'tests/matched_completion_v1').glob('*.py')), ROOT / '.gitattributes']
    freeze = {'version':P['version'], 'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
              'old_source_sha256':oldsources, 'old_result_and_input_sha256':bound,
              'software_qualification_sha256':sha(OUTPUT / 'software/qualification.json'),
              'frozen_before_new_fits':True, 'new_fits_at_freeze':0,
              'protocol_sha256':sha(PROTOCOL_PATH), 'actual_cwd':str(Path.cwd()),
              'actual_import':str(ROOT / 'src/wphid/__init__.py'), 'output_root':str(OUTPUT)}
    save(OUTPUT / 'source-freeze.json',freeze)
    save(OUTPUT / 'development-inputs.json',{'rows':rows,'source_freeze_sha256':sha(OUTPUT / 'source-freeze.json'),
                                          'new_noise_packets':0,'new_trajectories':0})
    print(json.dumps({'prepared':True,'old_wphid_reuses':18,'new_adam_jobs':18}),flush=True)

def initializer():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify_freeze()

def job_path(job):
    return OUTPUT / 'fits' / job['stage'] / (job['id']+'.json')

def validate_record(record, job):
    if (record['job'] != job or record['input_sha256'] != job['input']['sha256']
            or record['source_freeze_sha256'] != sha(OUTPUT / 'source-freeze.json') or not record['completed']):
        raise ValueError('Completed fit identity mismatch')

def run_job(job):
    path = job_path(job)
    if path.exists():
        previous = read(path)
        validate_record(previous,job)
        return {'id':job['id'],'reused':True,'status':previous['status']}
    from scientific_io import load_conditions
    from wphid.final_candidate_v1.problem import paper_families
    from wphid.final_candidate_v1.solver import Candidate, Mechanism, FitBudget, fit_candidate
    from wphid.final_candidate_v1.pipeline import prepare_views, fit_record, frozen_residual_report
    from wphid.final_candidate_v1.exact_gradient import ExactGradientView
    from wphid.adam_known_support_v2 import KnownSupportBudget, fit_known_support
    started_cpu, started_wall = time.process_time(),time.perf_counter()
    conditions = load_conditions(ROOT / job['input']['file'],job['input']['sha256'])
    family = next(f for f in paper_families() if f.name == ORIGINAL['Q1']['known_family'])
    candidate = Candidate((Mechanism('reaction',family,tuple(ORIGINAL['Q1']['known_partition_by_role'][job['role']])),))
    training,heldout = prepare_views(conditions,(family,),tuple(ORIGINAL['train_conditions']),tuple(ORIGINAL['heldout_conditions']))
    record = {'job':job,'input_sha256':job['input']['sha256'],
              'source_freeze_sha256':sha(OUTPUT / 'source-freeze.json'),
              'completed':True,'actual_import':str(ROOT / 'src/wphid/__init__.py'),
              'training_scale_sha256':training.scale.fingerprint,'truth_used_in_fitting':False,
              'heldout_used_in_fitting':False,'threads':1,'dtype':'float64','device':'CPU',
              'shared_input_loading_and_scaling_cpu_seconds':time.process_time()-started_cpu}
    try:
        if job['program'] == 'adam':
            settings = ORIGINAL['adam_budget']
            starts = FitBudget(seeds=tuple(settings['seeds']),initial_fractions=tuple(settings['initial_fractions']),maximum_cpu_seconds=settings['maximum_cpu_seconds'])
            budget = KnownSupportBudget(starts=starts,**{k:settings[k] for k in ('epochs_per_start','learning_rate','scheduler_step','scheduler_multiplier','maximum_backward_calls')})
            result = fit_known_support(ExactGradientView(training.conditions,training.scale),candidate,budget)
            fitted = result.pop('fit')
            record['adam_adaptation'] = result
        else:
            settings = dict(ORIGINAL['wphid_budget'])
            for k in ('seeds','initial_fractions'):
                settings[k] = tuple(settings[k])
            budget = FitBudget(**settings)
            fitted = fit_candidate(training,candidate,budget)
        record.update({'status':'success','budget':asdict(budget),'fit':fit_record(training,fitted),
                       'flat_parameters':fitted.parameters.tolist(),'frozen_heldout':frozen_residual_report(heldout,fitted)})
    except (RuntimeError,FloatingPointError,ValueError) as error:
        record.update({'status':'failed','error_type':type(error).__name__,'error':str(error),'failure_retained_in_denominator':True})
    record.update({'total_worker_job_cpu_seconds':time.process_time()-started_cpu,
                   'total_worker_job_wall_seconds':time.perf_counter()-started_wall,
                   'cpu_cap_guarantees_convergence':False,'post_fit_profile_refinement':False})
    save(path,record)
    return {'id':job['id'],'reused':False,'status':record['status'],'cpu_seconds':record['total_worker_job_cpu_seconds']}

def jobs(stage, packets):
    return [{'id':f"{p['seed']}_{p['role']}_{program}",'stage':stage,'role':p['role'],'program':program,'input':p}
            for p in packets for program in (('adam',) if stage == 'development' else ('wphid','adam'))]

def execute(stage):
    verify_freeze()
    complete = OUTPUT / (stage+'-execution-complete.json')
    if complete.exists():
        raise FileExistsError('Stage complete; read saved records, never rerun')
    data = read(OUTPUT / (stage+'-inputs.json'))
    packets = [r['input'] for r in data['rows']] if stage == 'development' else data['records']
    declared = jobs(stage,packets)
    started,finished = time.perf_counter(),[]
    with ProcessPoolExecutor(max_workers=P['worker_processes'],initializer=initializer) as executor:
        futures = [executor.submit(run_job,j) for j in declared]
        for future in as_completed(futures):
            result = future.result()
            finished.append(result)
            save(OUTPUT / (stage+'-status.json'),{'stage':stage,'completed':len(finished),'total':len(declared),'last':result,'wall_seconds':time.perf_counter()-started},exclusive=False)
            print(json.dumps({'stage':stage,'completed':len(finished),'total':len(declared),'last':result}),flush=True)
    verify_freeze()
    receipts = {job_path(j).relative_to(ROOT).as_posix():sha(job_path(j)) for j in declared}
    save(complete,{'version':P['version'],'stage':stage,'fit_sha256':receipts,
                   'source_freeze_sha256':sha(OUTPUT / 'source-freeze.json'),'inputs_sha256':sha(OUTPUT / (stage+'-inputs.json')),
                   'new_fits':sum(not r['reused'] for r in finished),'completed_failures_retained':sum(r['status']=='failed' for r in finished),
                   'worker_job_cpu_seconds':sum(read(job_path(j))['total_worker_job_cpu_seconds'] for j in declared),
                   'wall_seconds':time.perf_counter()-started,'truth_accuracy_assessed':False})

def computability_gate(rows):
    counts = {role:sum(r['assessable'] for r in rows if r['role']==role) for role in P['roles']}
    failures = sum(not r['success'] for r in rows)
    return {'passed':all(n >= P['development_gate']['minimum_assessable_per_role'] for n in counts.values()) and failures==0,
            'assessable_per_role':counts,'thrown_numeric_model_failures':failures,'precision_is_separate':True}

def assess(stage):
    verify_freeze()
    completion = read(OUTPUT / (stage+'-execution-complete.json'))
    verify_hashes(completion['fit_sha256'])
    if completion['source_freeze_sha256'] != sha(OUTPUT / 'source-freeze.json'):
        raise ValueError('Execution source differs')
    # Offline truth import occurs only after every immutable scientific fit exists.
    from assess_confirmation import assess_fit, descriptive
    data = read(OUTPUT / (stage+'-inputs.json'))
    packets = [r['input'] for r in data['rows']] if stage=='development' else data['records']
    rows,lookup = [],{}
    for packet in packets:
        for program in ('wphid','adam'):
            if stage=='development' and program=='wphid':
                receipt = next(r for r in data['rows'] if r['input']==packet)
                path = ROOT / receipt['old_wphid_file']
                if sha(path)!=receipt['old_wphid_sha256']:
                    raise ValueError('Old WPHID reuse changed')
                record=read(path)
            else:
                job=next(j for j in jobs(stage,[packet]) if j['program']==program)
                record=read(job_path(job))
                validate_record(record,job)
            row={'seed':packet['seed'],'program':program,**assess_fit(record,packet['role'])}
            rows.append(row)
            lookup[packet['seed'],packet['role'],program]=row
    summary={role:{program:descriptive([r for r in rows if r['role']==role and r['program']==program]) for program in ('wphid','adam')} for role in P['roles']}
    blocks=[]
    for seed in sorted({p['seed'] for p in packets}):
        scores={}
        for program in ('wphid','adam'):
            selected=[lookup[seed,role,program] for role in P['roles']]
            scores[program]=max(r['case_score'] for r in selected) if all(r['assessable'] for r in selected) else None
        valid=all(v is not None for v in scores.values())
        blocks.append({'seed':seed,'scores':scores,'both_assessable':valid,'wphid_win':bool(valid and scores['adam']>0 and scores['wphid']<=.9*scores['adam'])})
    result={'version':P['version'],'stage':stage,'rows':rows,'role_summary':summary,'blocks':blocks,
            'execution_receipt_sha256':sha(OUTPUT / (stage+'-execution-complete.json')),
            'source_freeze_sha256':sha(OUTPUT / 'source-freeze.json'),'native_ADAM_SINDy':False}
    if stage=='development':
        result['computability_gate']=computability_gate([r for r in rows if r['program']=='adam'])
        result['dependent_fresh_confirmation_allowed']=result['computability_gate']['passed']
        result['development_not_independent_confirmation']=True
    else:
        from scipy.stats import binomtest
        wins=sum(b['wphid_win'] for b in blocks)
        rates=all(s['wphid']['precision_passes']>=s['adam']['precision_passes'] for s in summary.values())
        result['formal_conditional_advantage']={'wphid_wins':wins,'denominator':18,'one_sided_binomial_p':float(binomtest(wins,18,.5,alternative='greater').pvalue),
            'required_wins':13,'precision_rate_gate':rates,'passed':wins>=13 and rates,
            'not_a_full_discovery_or_engineering_adoption_gate':True}
    save(OUTPUT / (stage+'-assessment.json'),result)
    print(json.dumps({'stage':stage,'role_summary':summary,'gate':result.get('computability_gate',result.get('formal_conditional_advantage'))}),flush=True)

def generate_fresh(seed,role):
    from scientific_io import CONDITIONS
    from wphid.final_candidate_v1.observations import adapt_condition
    from experiments import paper_h3_joint_family_topology_partition as reference
    import numpy as np
    namespace=P['fresh_confirmation']['namespace']
    destination=OUTPUT / 'inputs/fresh' / f'{seed}_{role}.npz'
    if destination.exists():
        record=read(destination.with_suffix('.json'))
        if (record['seed']!=seed or record['role']!=role or record['namespace']!=namespace or sha(destination)!=record['sha256']):
            raise ValueError('Existing new input differs')
        return record
    word=int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:4],'little')
    arrays,clipping={},{}
    for index,name in enumerate(CONDITIONS):
        t,z,r,clean,forcing=reference._clean(name,role)
        rng=np.random.default_rng(np.random.SeedSequence((seed,index,word)))
        observed=[s+rng.normal(0.,.0005*max(float(np.sqrt(np.mean(s*s))),1e-12),s.shape) for s in clean]
        clipping[name]={'oxygen':int(np.count_nonzero(observed[1]<1e-4)),'damage':int(np.count_nonzero((observed[2]<1e-8)|(observed[2]>.95)))}
        observed[1]=np.maximum(observed[1],1e-4)
        observed[2]=np.clip(observed[2],1e-8,.95)
        condition=adapt_condition(name=name,t=t,z=z,r=r,observed_states=tuple(observed),supplied_forcing=forcing)
        arrays.update({name+'_states':np.stack(observed),name+'_forcing':np.stack(forcing),name+'_bases':np.stack(condition.bases),name+'_targets':np.stack(condition.targets)})
        if index==0:
            arrays.update({'t':t,'z':z,'r':r,'projection_data':condition.projection.data,'projection_indices':condition.projection.indices,'projection_indptr':condition.projection.indptr,'projection_shape':np.asarray(condition.projection.shape)})
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    record={'seed':seed,'role':role,'namespace':namespace,'file':destination.relative_to(ROOT).as_posix(),'sha256':sha(destination),
            'conditions':list(CONDITIONS),'grid_shape':[25,25,25],'noise_fraction_clean_field_rms':.0005,'clipping':clipping,
            'new_physical_trajectory_integrations':0,'common_field_noise_across_roles_in_same_seed':True,'parameters_not_present_in_estimator_input':True}
    save(destination.with_suffix('.json'),record)
    return record

def prepare_fresh():
    verify_freeze()
    gate=read(OUTPUT / 'development-assessment.json')
    if not gate['computability_gate']['passed']:
        raise ValueError('Dependent confirmation gate failed; do not generate new inputs')
    if (OUTPUT / 'fresh-inputs.json').exists():
        raise FileExistsError('Fresh inputs frozen; reuse never regenerate')
    records=[generate_fresh(seed,role) for seed in P['fresh_confirmation']['seeds'] for role in P['roles']]
    save(OUTPUT / 'fresh-inputs.json',{'records':records,'source_freeze_sha256':sha(OUTPUT / 'source-freeze.json'),
        'development_assessment_sha256':sha(OUTPUT / 'development-assessment.json'),'all_inputs_frozen_before_fresh_fits':True,
        'new_noise_packets':54,'independent_blocks':18,'new_physical_trajectory_integrations':0})
    print(json.dumps({'fresh_packets':54,'fresh_paired_fits':108}),flush=True)

if __name__=='__main__':
    command=sys.argv[1]
    if command=='prepare': prepare()
    elif command=='prepare-fresh': prepare_fresh()
    elif command.startswith('run-'): execute(command[4:])
    elif command.startswith('assess-'): assess(command[7:])
    elif command=='verify': verify_freeze(); print('freeze verified',flush=True)
    else: raise ValueError('Unknown command')
