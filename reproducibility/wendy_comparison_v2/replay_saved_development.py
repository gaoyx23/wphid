"""Read-only replay of immutable Gaussian development points, without fits."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(ROOT/'reproducibility/wendy_comparison_v1'))
from run_fixed_comparison import read,sha,save,verify,PROTOCOL,load_conditions,prepare_views,paper_families,weak_operators
from wphid.wendy_comparison_v1 import GaussianWeakProblem
import numpy as np

def main():
    output=ROOT/read(PROTOCOL)['output_root']
    frozen=verify(output)
    complete=read(output/'execution-complete.json')
    protocol=read(PROTOCOL)
    rows=[]
    for job in frozen['jobs']:
        path=output/f'fits/{job["id"]}.json'
        if sha(path)!=complete['fit_sha256'][job['id']]:
            raise ValueError('Saved fit changed')
        record=read(path);fit=record['fit']
        if fit['parameters'] is None:
            rows.append({'id':job['id'],'status':'no_finite_point_to_replay'})
            continue
        conditions=load_conditions(ROOT/job['input_file'],job['input_sha256'])
        training,_=prepare_views(conditions,paper_families(),protocol['training'],protocol['heldout'])
        family=next(f for f in paper_families() if f.name=='arrhenius_oxygen_damage')
        with np.load(ROOT/job['input_file'],allow_pickle=False) as data:
            operators=weak_operators(data['t'],data['z'],data['r'],tuple(protocol['window_shape']),tuple(protocol['stride_shape']))
        problem=GaussianWeakProblem(training,family,job['groups'],operators,record['sigmas_observed_training'])
        unit=(np.asarray(fit['parameters'])-problem.lower)/problem.span
        vector=np.r_[unit,np.asarray(fit['scaled_coefficients']).ravel()]
        value,gradient=problem.value_gradient(vector)
        objective_error=abs(value-fit['quasi_negative_log_likelihood'])/max(1.,abs(fit['quasi_negative_log_likelihood']))
        physical_error=float(np.max(np.abs(np.asarray(problem.physical_coefficients(vector))-fit['physical_coefficients'])/
                                     np.maximum(1.,np.abs(fit['physical_coefficients']))))
        row={'id':job['id'],'relative_objective_replay_error':objective_error,
             'relative_physical_normalization_error':physical_error,'gradient_finite':bool(np.isfinite(gradient).all()),
             'passed':objective_error<=1e-9 and physical_error<=1e-12 and bool(np.isfinite(gradient).all())}
        rows.append(row)
        print(json.dumps(row),flush=True)
    result={'rows':rows,'replayed_points':sum('passed' in r for r in rows),'passed':sum(r.get('passed',False) for r in rows),
            'source_freeze_sha256':sha(output/'source-freeze.json'),'execution_complete_sha256':sha(output/'execution-complete.json'),
            'audit_source_sha256':sha(Path(__file__)),'new_parameter_fits':0,'new_noise_draws':0,'new_physical_trajectories':0,
            'score_based_point_selection':False,'parameter_accuracy_claim':False}
    save(output/'saved-fit-replay-audit.json',result)

if __name__=='__main__':
    main()
