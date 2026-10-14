from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fresh_gaussian_wrapper',ROOT/'reproducibility/wendy_comparison_v2/run_fresh_conditional.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def packets():
    protocol=module.read(module.PROTO)
    values=[{'seed':s,'role':r,'file':f'{s}_{r}.npz','sha256':str(s),
             'parameters_not_present_in_estimator_input':True,'clipping':{'example':{'oxygen':0,'damage':0}}}
            for s in protocol['fresh_blocks'] for r in protocol['roles']]
    return values,protocol

def test_exact_fresh_packets_and_no_control_or_truth_parameters_in_jobs():
    values,protocol=packets();jobs=module.declared_jobs(values,protocol)
    assert len(jobs)==54
    assert all(set(j)=={'id','seed','role','groups','input_file','input_sha256'} for j in jobs)
    values.append(deepcopy(values[0]))
    with pytest.raises(ValueError,match='Exactly'):
        module.declared_jobs(values,protocol)

def test_clipped_or_nonqualified_packets_are_rejected():
    values,protocol=packets();values[0]['clipping']['example']['oxygen']=1
    with pytest.raises(ValueError,match='clipping'):
        module.declared_jobs(values,protocol)
    values,protocol=packets();values[0]['parameters_not_present_in_estimator_input']=False
    with pytest.raises(ValueError,match='truth-free'):
        module.declared_jobs(values,protocol)

def test_control_fit_identity_requires_exact_input_source_job_and_completion():
    values,protocol=packets();job=module.declared_jobs(values,protocol)[0]
    record={'completed':True,'source_freeze_sha256':'source','input_sha256':job['input_sha256'],
            'job':{'id':job['id']+'_wphid','program':'wphid','role':job['role'],'stage':'fresh','input':{'sha256':job['input_sha256']}}}
    module.validate_control(record,job,'wphid','source')
    for key in ('completed','source_freeze_sha256','input_sha256'):
        altered=deepcopy(record);altered[key]=None
        with pytest.raises(ValueError,match='identity'):
            module.validate_control(altered,job,'wphid','source')
    altered=deepcopy(record);altered['job']['id']='another_job'
    with pytest.raises(ValueError,match='identity'):
        module.validate_control(altered,job,'wphid','source')
