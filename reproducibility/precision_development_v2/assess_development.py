"""Replay and grade only the completed new development; reuse saved V1 grades."""

import csv
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from development_io import ROOT,OUTPUT,PRIOR,PROTOCOL,read,sha,write_new,verify_freeze


def main():
    manifest=verify_freeze()
    if (OUTPUT/'assessment.json').exists():raise FileExistsError('Saved development assessment must not be overwritten')
    completed=read(OUTPUT/'execution-complete.json')
    if (completed['completed_jobs']!=PROTOCOL['new_estimates'] or completed['source_freeze_sha256']!=sha(OUTPUT/'source-freeze.json')
            or completed['input_manifest_sha256']!=sha(OUTPUT/'input-manifest.json')):
        raise ValueError('Not all fixed development estimates completed under this freeze')
    records=[]
    for job in manifest['jobs']:
        path=OUTPUT/'fits'/(job['id']+'.json');record=read(path)
        if sha(path)!=completed['new_fit_source_sha256'][path.relative_to(ROOT).as_posix()] or record['job']!=job:
            raise ValueError('New fit identity differs from completed record')
        records.append(record)
    # Original assessor is imported only after all 12 immutable new records exist.
    sys.path.insert(1,str(ROOT/'reproducibility/final_candidate_v1'))
    sys.path.insert(2,str(ROOT))
    from assess_confirmation import assess_fit,restore_fit,descriptive
    from scientific_io import load_conditions
    from wphid.final_candidate_v1.problem import paper_families
    from wphid.final_candidate_v1.pipeline import prepare_views,frozen_residual_report
    from wphid.final_candidate_v1.solver import design,physical_coefficients,ScoreConvention
    import numpy as np
    prior_assessment=read(PRIOR/'assessment.json')
    prior_rows={(r['seed'],r['role']):r for r in prior_assessment['Q1']['rows'] if r['program']=='wphid'}
    pairs=[];replays=[]
    family=next(f for f in paper_families() if f.name==PROTOCOL['known_family'])
    for record in records:
        job=record['job'];old=read(ROOT/job['V1_fit']);original_grade=prior_rows[job['seed'],job['role']]
        passed=False
        if record['status']=='success':
            conditions=load_conditions(ROOT/job['input']['file'],job['input']['sha256'])
            training,heldout=prepare_views(conditions,(family,),tuple(PROTOCOL['train_conditions']),tuple(PROTOCOL['heldout_conditions']))
            fit=restore_fit(record)
            matrices=design(training,fit.candidate,fit.parameters)
            rss=tuple(float((y-a@c)@(y-a@c)) for y,a,c in zip(training.targets,matrices,fit.coefficients))
            np.testing.assert_allclose(rss,fit.equation_rss,rtol=5e-8,atol=1e-10)
            convention=ScoreConvention.from_targets(training.targets)
            np.testing.assert_allclose(convention.objective(rss),fit.training_objective,rtol=5e-8,atol=1e-10)
            np.testing.assert_allclose(convention.score(rss,fit.nominal_parameter_count),fit.final_score,rtol=5e-8,atol=1e-10)
            for computed,saved in zip(physical_coefficients(training,fit),record['fit']['physical_coefficients']):
                np.testing.assert_allclose(computed,saved,rtol=5e-8,atol=1e-10)
            report=frozen_residual_report(heldout,fit)
            np.testing.assert_allclose(report['equation_weak_target_relative_residual'],
                                       record['frozen_heldout']['equation_weak_target_relative_residual'],rtol=5e-8,atol=1e-10)
            passed=True
        grade=assess_fit(record,job['role'])
        ratio=(grade['case_score']/original_grade['case_score'] if grade['assessable']
               and original_grade['assessable'] and original_grade['case_score']>0 else None)
        delta=(old['fit']['training_objective']-record['fit']['training_objective'] if record['status']=='success' else None)
        pairs.append({'seed':job['seed'],'role':job['role'],'V1_saved_grade':original_grade,'V2_grade':grade,
                      'estimate_source':'new_changed_path_development',
                      'case_score_ratio_V2_over_V1':ratio,'training_objective_decrease_V1_minus_V2':delta,
                      'V1_CPU_seconds':old['fit']['cpu_seconds'],'V2_CPU_seconds':record.get('fit',{}).get('cpu_seconds'),
                      'new_model_replay_passed':passed,'component_audit':record.get('component_solver_audit')})
        replays.append({'id':job['id'],'fit_saved':record['status']=='success','model_replay_passed':passed})
    for control in manifest['unchanged_controls']:
        original_grade=prior_rows[control['seed'],control['role']]
        old=read(ROOT/control['V1_fit'])
        pairs.append({'seed':control['seed'],'role':control['role'],'V1_saved_grade':original_grade,
                      'V2_grade':original_grade,'estimate_source':'unchanged_V1_control_reused',
                      'case_score_ratio_V2_over_V1':1.,'training_objective_decrease_V1_minus_V2':0.,
                      'V1_CPU_seconds':old['fit']['cpu_seconds'],'V2_CPU_seconds':None,
                      'new_model_replay_passed':None,'component_audit':None})
    summaries={}
    for role in PROTOCOL['roles']:
        selected=[p for p in pairs if p['role']==role]
        ratios=[p['case_score_ratio_V2_over_V1'] for p in selected if p['case_score_ratio_V2_over_V1'] is not None]
        deltas=[p['training_objective_decrease_V1_minus_V2'] for p in selected if p['training_objective_decrease_V1_minus_V2'] is not None]
        summaries[role]={'V1':descriptive([p['V1_saved_grade'] for p in selected]),
                         'V2':descriptive([p['V2_grade'] for p in selected]),
                         'median_paired_case_score_ratio':float(np.median(ratios)) if ratios else None,
                         'median_training_objective_decrease':float(np.median(deltas)) if deltas else None,
                         'training_objective_decrease_at_least_one':sum(d>=1. for d in deltas),
                         'case_score_improved':sum(r<1. for r in ratios),'case_denominator':6}
        if role in PROTOCOL['unchanged_control_roles']:
            summaries[role]['V2']['median_cpu_seconds']=None
            summaries[role]['V2']['estimate_source']='unchanged_V1_controls_reused; no new timing'
    valid=sum(p['V2_grade']['assessable'] for p in pairs)
    precision_gate=(valid==18 and summaries['shared']['V2']['precision_passes']==6
                    and all(summaries[r]['V2']['precision_passes']>=5
                            and summaries[r]['median_paired_case_score_ratio'] is not None
                            and summaries[r]['median_paired_case_score_ratio']<=.8 for r in ('partial','independent')))
    replay_gate=all(r['model_replay_passed'] for r in replays if r['fit_saved'])
    worker_gate=not completed['worker_guard_violations']
    combined_CPU=(PROTOCOL['previous_saved_confirmation_worker_CPU_seconds']+
                  PROTOCOL['previous_interrupted_unsaved_CPU_upper_seconds']+completed['saved_worker_CPU_seconds'])
    budget_gate=combined_CPU<=3600*PROTOCOL['combined_worker_CPU_budget_hours']
    result={'version':PROTOCOL['version'],'role':'development on previously observed six blocks, not independent confirmation',
            'source_freeze_sha256':sha(OUTPUT/'source-freeze.json'),'input_manifest_sha256':sha(OUTPUT/'input-manifest.json'),
            'execution_complete_sha256':sha(OUTPUT/'execution-complete.json'),'new_fit_source_sha256':completed['new_fit_source_sha256'],
            'pairs':pairs,'role_summary':summaries,'new_estimates':12,'new_estimates_assessable':valid-6,'reused_V1_estimates':18,
            'reused_unchanged_shared_controls':6,'new_observations':0,
            'all_saved_model_replays':replay_gate,'successful_model_replays':sum(r['model_replay_passed'] for r in replays),
            'replay_optimizer_calls':0,'replay_coefficient_refits':0,'precision_development_gate':precision_gate,
            'worker_budget_gate':worker_gate and budget_gate,'development_adoption_gate':precision_gate and replay_gate and worker_gate and budget_gate,
            'saved_development_worker_CPU_seconds':completed['saved_worker_CPU_seconds'],
            'combined_saved_worker_CPU_seconds_plus_interruption_reserve':combined_CPU,
            'independent_confirmation':False,'old_grades_reused_not_rescored':True,'old_results_modified':False,
            'remaining':'No automatic additional fits or confirmation; an optimization change cannot be called a precision repair without passing the declared gate.'}
    write_new(OUTPUT/'assessment.json',result)
    write_new(OUTPUT/'saved-model-replay.json',{'records':replays,'optimizer_calls':0,'coefficient_refits':0,
                                             'all_successful_models_pass':replay_gate})
    csv_path=OUTPUT/'paired-development.csv'
    with csv_path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','role','V1_precision_pass','V2_precision_pass','V1_case_score','V2_case_score',
                                                 'case_score_ratio','training_objective_decrease','V1_CPU_seconds','V2_CPU_seconds'])
        writer.writeheader()
        for p in pairs:
            writer.writerow({'seed':p['seed'],'role':p['role'],'V1_precision_pass':p['V1_saved_grade']['precision_pass'],
                             'V2_precision_pass':p['V2_grade']['precision_pass'],'V1_case_score':p['V1_saved_grade']['case_score'],
                             'V2_case_score':p['V2_grade']['case_score'],'case_score_ratio':p['case_score_ratio_V2_over_V1'],
                             'training_objective_decrease':p['training_objective_decrease_V1_minus_V2'],
                             'V1_CPU_seconds':p['V1_CPU_seconds'],'V2_CPU_seconds':p['V2_CPU_seconds']})
    print({'development_adoption_gate':result['development_adoption_gate'],'role_summary':summaries,
           'saved_worker_CPU_hours':completed['saved_worker_CPU_seconds']/3600.,'new_estimates':12,'reused_shared_controls':6,'new_observations':0})


if __name__=='__main__':main()
