"""Independent saved-point replay: physical prediction RSS -> fitting units.

The V2 estimator and its already frozen first replay are preserved. This
postprocessing does not fit or select any parameter or observation.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(1,str(ROOT))
from reproducibility.matched_completion_resume_v2 import complete as c
import numpy as np

def fitting_rss(targets,predictions,target_weights):
    return np.asarray([np.sum((y-p)**2)*weight**2 for y,p,weight in zip(targets,predictions,target_weights)])

def main():
    c.verify()
    qualification=c.w.read(c.OUTPUT/'replay-r1-software/qualification.json')
    if not qualification['all_passed']:raise ValueError('Replay unit qualification missing')
    c.w.verify_hashes(qualification['source_sha256'])
    completion=c.w.read(c.OUTPUT/'fresh-execution-complete.json')
    c.w.verify_hashes(completion['fit_sha256'])
    if not (c.OUTPUT/'fresh-assessment.json').exists():raise ValueError('Original108 assessment required first')
    from scientific_io import load_conditions
    from assess_confirmation import restore_fit
    from wphid.final_candidate_v1.problem import paper_families
    from wphid.final_candidate_v1.pipeline import prepare_views,frozen_residual_report
    from wphid.final_candidate_v1.solver import predict_frozen,physical_coefficients
    rows=[];bound={}
    for job in c.w.jobs('fresh',c.w.read(c.OUTPUT/'fresh-inputs.json')['records']):
        path=c.w.job_path(job);record=c.w.read(path);c.w.validate_record(record,job);bound[path.relative_to(ROOT).as_posix()]=c.w.sha(path)
        if record['status']!='success':
            rows.append({'id':job['id'],'replayed':False,'saved_failure_retained':True});continue
        conditions=load_conditions(ROOT/job['input']['file'],job['input']['sha256'])
        family=next(f for f in paper_families() if f.name==c.w.ORIGINAL['Q1']['known_family'])
        training,heldout=prepare_views(conditions,(family,),tuple(c.w.ORIGINAL['train_conditions']),tuple(c.w.ORIGINAL['heldout_conditions']))
        fit=restore_fit(record)
        if training.scale.fingerprint!=record['training_scale_sha256']:raise ValueError('Frozen scale differs')
        predictions=predict_frozen(training,fit)
        targets=tuple(np.concatenate([condition.targets[e] for condition in training.conditions]) for e in range(len(predictions)))
        rss=fitting_rss(targets,predictions,training.scale.target_weights)
        np.testing.assert_allclose(rss,record['fit']['equation_training_rss'],rtol=2e-8,atol=2e-10)
        for left,right in zip(physical_coefficients(training,fit),record['fit']['physical_coefficients']):
            np.testing.assert_allclose(left,right,rtol=3e-13,atol=1e-13)
        held=frozen_residual_report(heldout,fit)
        np.testing.assert_allclose(held['equation_weak_target_relative_residual'],record['frozen_heldout']['equation_weak_target_relative_residual'],rtol=3e-13,atol=1e-13)
        rows.append({'id':job['id'],'replayed':True,'maximum_absolute_fitting_RSS_difference':float(np.max(np.abs(rss-np.asarray(record['fit']['equation_training_rss'])))),
                     'heldout_refit':False,'new_fit_or_selection':False})
    c.verify();c.w.verify_hashes(bound)
    c.save(c.OUTPUT/'saved-point-replay-r1.json',{'version':'PAIRED_SAVED_POINT_REPLAY_UNITS_R1_20261013','rows':rows,
           'declared':108,'replayed_successes':sum(r['replayed'] for r in rows),'saved_failures_retained':sum(r.get('saved_failure_retained',False) for r in rows),
           'fit_sha256':bound,'units':'predict_frozen physical RSS multiplied by each frozen target_weight squared',
           'qualification_sha256':c.w.sha(c.OUTPUT/'replay-r1-software/qualification.json'),
           'original_scientific_source_freeze_sha256':c.w.sha(c.OUTPUT/'recovery-v2-source-freeze.json'),
           'new_fits_noise_selection':0,'whole_batch_successful_replay_counted_once':True,'recorded_utc':c.r1.now()})
    stderr=c.OUTPUT/'detached.stderr.log';error=stderr.read_text(encoding='utf-8',errors='replace') if stderr.exists() else ''
    plan=c.w.read(c.OUTPUT/'recovery-v2-plan.json')
    c.save(c.OUTPUT/'administrative-assessment-context-r1.json',{'version':c.P['version'],
           'fresh_assessment_sha256':c.w.sha(c.OUTPUT/'fresh-assessment.json'),
           'replay_sha256':c.w.sha(c.OUTPUT/'saved-point-replay-r1.json'),
           'source_freeze_sha256':c.w.sha(c.OUTPUT/'recovery-v2-source-freeze.json'),
           'old_completed_byte_reuses':105,'new_retained_attempts':3,'one_complete_scored_attempt_per_job':True,
           'old_lost_points_unavailable_not_selected':True,'old105_unchanged':True,'original_gates_and_kernels_unchanged':True,
           'candidate_total600_budget_unchanged_claim':False,'lost_CPU_second_interruption':plan['lost_CPU'],
           'lost_CPU_first_interruption_nominal_interval_seconds':[0,2400],'pure_same_loss_optimizer_comparison':False,
           'initial_V2_replay_units_issue_detected_statically_before_execution':True,
           'actual_initial_replay_AssertionError_in_preserved_log':'AssertionError' in error,
           'initial_replay_stderr_sha256':c.w.sha(stderr) if stderr.exists() else None,
           'independent_units_replay_changes_no_estimate':True,'new_noise_or_integrations':0,'whole_method_adoption':False,
           'recorded_utc':c.r1.now()})
    print(json.dumps({'successful_saved_point_replay':sum(r['replayed'] for r in rows),'declared':108,'new_scientific_fits':0}),flush=True)

if __name__=='__main__':main()
