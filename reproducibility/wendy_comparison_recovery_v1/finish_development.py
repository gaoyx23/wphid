"""Replay and apply retained pre-scoring gates to the recovered18 points."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(Path(__file__).parent))
from workflow import read,sha,save,verify_recovery,OUTPUT,ORIGINAL,P
from run_fixed_comparison import load_conditions,prepare_views,paper_families,weak_operators
from wphid.wendy_comparison_v1 import GaussianWeakProblem
import numpy as np

def replay():
    recovery=verify_recovery()
    completion=read(OUTPUT/'execution-complete.json')
    protocol=read(ROOT/'reproducibility/wendy_comparison_v1/scientific-protocol-20261013.json')
    original=read(OUTPUT/'source-freeze.json')
    rows=[]
    for job in original['jobs']:
        path=OUTPUT/f'fits/{job["id"]}.json'
        if sha(path)!=completion['fit_sha256'][job['id']]:
            raise ValueError('Selected saved point changed')
        record=read(path);fit=record['fit']
        if fit['parameters'] is None:
            rows.append({'id':job['id'],'replayed':False,'passed':False});continue
        conditions=load_conditions(ROOT/job['input_file'],job['input_sha256'])
        training,_=prepare_views(conditions,paper_families(),protocol['training'],protocol['heldout'])
        family=next(f for f in paper_families() if f.name=='arrhenius_oxygen_damage')
        with np.load(ROOT/job['input_file'],allow_pickle=False) as data:
            operators=weak_operators(data['t'],data['z'],data['r'],tuple(protocol['window_shape']),tuple(protocol['stride_shape']))
        problem=GaussianWeakProblem(training,family,job['groups'],operators,record['sigmas_observed_training'])
        vector=np.r_[(np.asarray(fit['parameters'])-problem.lower)/problem.span,np.asarray(fit['scaled_coefficients']).ravel()]
        value,gradient=problem.value_gradient(vector)
        objective_error=abs(value-fit['quasi_negative_log_likelihood'])/max(1.,abs(fit['quasi_negative_log_likelihood']))
        physical_error=float(np.max(np.abs(np.asarray(problem.physical_coefficients(vector))-fit['physical_coefficients'])/
                                    np.maximum(1.,np.abs(fit['physical_coefficients']))))
        row={'id':job['id'],'replayed':True,'relative_objective_error':objective_error,
             'relative_physical_coefficient_error':physical_error,'gradient_finite':bool(np.isfinite(gradient).all()),
             'passed':objective_error<=1e-9 and physical_error<=1e-12 and bool(np.isfinite(gradient).all())}
        rows.append(row);print(json.dumps(row),flush=True)
    save(OUTPUT/'saved-fit-replay-audit.json',{'rows':rows,'passed':sum(r['passed'] for r in rows),'replayed':sum(r['replayed'] for r in rows),
         'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),'recovery_source_freeze_sha256':sha(OUTPUT/'recovery-source-freeze.json'),
         'execution_complete_sha256':sha(OUTPUT/'execution-complete.json'),'audit_source_sha256':sha(Path(__file__)),
         'new_parameter_fits':0,'new_noise_draws':0,'new_trajectory_integrations':0,'score_based_point_selection':False})

def decide():
    verify_recovery()
    declaration=ROOT/'reproducibility/wendy_comparison_v2/conditional-independent-protocol-20261013.json'
    directional=ROOT/'reproducibility/wendy_comparison_v2/independent-directional-family-clarification-r1-20261013.json'
    controls=ROOT/'reproducibility/wendy_comparison_v2/control-identity-freeze-clarification-r2-20261013.json'
    expected={declaration:'955ee9830530c031c81b97e4a8ad24a4684b5729be205459d6480ccf4a022a9e',
              directional:'f195beb733718d2f3aeb02c321278913afe948dcb36c702c3a7ea5aa6ec1dc18',
              controls:'572bd9a710c65ec464e8217f012a0b9ecaf6222063785ac8a63d22d05ce26452'}
    if any(sha(path)!=wanted for path,wanted in expected.items()):
        raise ValueError('Retained pre-scoring declarations changed')
    completion=read(OUTPUT/'execution-complete.json')
    assessment=read(OUTPUT/'assessment.json')
    if completion['selected_case_denominator']!=18 or assessment['execution_complete_sha256']!=sha(OUTPUT/'execution-complete.json'):
        raise ValueError('Recovered development assessment not bound to all18selected fits')
    adam_file=ROOT/'artifacts/paper_comparison_resume_v1/paired_confirmation/development-assessment.json'
    adam=read(adam_file)
    checks={}
    for role,summary in assessment['summary'].items():
        median,control=summary['median_case_score'],summary['median_control_case_score']
        checks[role]={'all_six_assessable':summary['assessable']==6,'precision_at_least_four':summary['precision_pass']>=4,
                      'median_reduction_at_least_twenty_percent':median is not None and control is not None and control>0 and median<=.8*control,
                      'Gaussian_median_case_score':median,'WPHID_median_case_score':control}
        checks[role]['passed']=all(checks[role][k] for k in ('all_six_assessable','precision_at_least_four','median_reduction_at_least_twenty_percent'))
    gates={'Gaussian_computability':assessment['qualification_gate_pass'],
           'Gaussian_promising_role':any(r['passed'] for r in checks.values()),
           'corrected_ADAM_computability':adam['computability_gate']['passed']}
    passed=all(gates.values())
    save(OUTPUT/'independent-entry-decision.json',{'gates':gates,'role_checks':checks,'entered_fresh_Gaussian_comparison':passed,
         'fresh_Gaussian_fits_authorized_by_gate':54 if passed else 0,'new_Gaussian_fits_in_this_decision':0,
         'new_noise_draws':0,'new_trajectory_integrations':0,'failed_gates':[k for k,v in gates.items() if not v],
         'protocol_sha256':sha(declaration),'directional_clarification_sha256':sha(directional),'control_identity_clarification_sha256':sha(controls),
         'development_assessment_sha256':sha(OUTPUT/'assessment.json'),'Adam_development_assessment_sha256':sha(adam_file),
         'recovery_source_freeze_sha256':sha(OUTPUT/'recovery-source-freeze.json'),'decision_source_sha256':sha(Path(__file__)),
         'status':'ready_for_frozen_fresh_packets_and_source_qualification' if passed else 'not_entered_dependency_gate_failed',
         'official_WENDy':False,'main_open_search_adoption':False})
    print(json.dumps({'gates':gates,'role_checks':checks,'entered_fresh_Gaussian_comparison':passed}),flush=True)

def cost():
    verify_recovery()
    cases=[read(p) for p in (OUTPUT/'fits').glob('*.json')]
    lost=set(P['administratively_interrupted_jobs'])
    saved_cpu=sum(r['fit']['cpu_seconds'] for r in cases)
    recovery_cpu=sum(r['fit']['cpu_seconds'] for r in cases if r['job']['seed'] in P['pending_seed_blocks'])
    save(OUTPUT/'administrative-cost-audit.json',{'selected_case_denominator':18,'exact_completed_reuses':12,'new_retained_recovery_fits':6,
         'selected_saved_solver_cpu_seconds':saved_cpu,'new_retained_recovery_solver_cpu_seconds':recovery_cpu,
         'lost_work_nominal_cpu_allowance_interval_seconds':[0,1200],'lost_actual_cpu_unrecorded':True,
         'lost_between_call_overshoot_unrecorded_and_excluded_from_nominal_interval':True,
         'candidate_total_600_cpu_claim':False,'administratively_recovered_ids':sorted(lost),
         'numeric_algorithm_retry':False,'one_saved_scored_point_per_original_case':True,
         'per_selected_attempt_nominal_cap_seconds':600,'maximum_selected_actual_solver_cpu_seconds':max(r['fit']['cpu_seconds'] for r in cases),
         'external_CPU_ledger_sha256':sha(OUTPUT/'external-cpu-ledger.jsonl'),
         'retained_input_or_old_result_changes':False,'official_WENDy':False})

if __name__=='__main__':
    {'replay':replay,'decide':decide,'cost':cost}[sys.argv[1]]()
