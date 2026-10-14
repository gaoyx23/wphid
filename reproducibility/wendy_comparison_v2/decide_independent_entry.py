"""Apply the predeclared supplemental Gaussian entry gate after all blind fits."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    protocol=Path(__file__).with_name('conditional-independent-protocol-20261013.json')
    clarification=Path(__file__).with_name('independent-directional-family-clarification-r1-20261013.json')
    if sha(protocol)!='955ee9830530c031c81b97e4a8ad24a4684b5729be205459d6480ccf4a022a9e':
        raise ValueError('Independent predeclared protocol identity changed')
    if sha(clarification)!='f195beb733718d2f3aeb02c321278913afe948dcb36c702c3a7ea5aa6ec1dc18':
        raise ValueError('Directional family clarification changed')
    output=ROOT/read(protocol)['development_source']
    complete=read(output/'execution-complete.json')
    if complete['new_fits']!=18:
        raise ValueError('Development not fully closed')
    result=read(output/'assessment.json')
    if result['execution_complete_sha256']!=sha(output/'execution-complete.json'):
        raise ValueError('Development assessment not bound to complete fit batch')
    adam_file=ROOT/'artifacts/paper_matched_completion_v1/development-assessment.json'
    adam=read(adam_file)
    role_checks={}
    for role,summary in result['summary'].items():
        median=summary['median_case_score'];control=summary['median_control_case_score']
        role_checks[role]={'all_six_assessable':summary['assessable']==6,
                          'precision_at_least_four':summary['precision_pass']>=4,
                          'median_reduction_at_least_twenty_percent':median is not None and control is not None and control>0 and median<=.8*control,
                          'Gaussian_median_case_score':median,'WPHID_median_case_score':control}
        role_checks[role]['passed']=all(role_checks[role][k] for k in ('all_six_assessable','precision_at_least_four','median_reduction_at_least_twenty_percent'))
    gates={'Gaussian_computability':result['qualification_gate_pass'],
           'Gaussian_promising_role':any(r['passed'] for r in role_checks.values()),
           'corrected_ADAM_computability':adam['computability_gate']['passed']}
    entered=all(gates.values())
    record={'gates':gates,'role_checks':role_checks,'entered_fresh_Gaussian_comparison':entered,
            'fresh_Gaussian_fits_authorized_by_gate':54 if entered else 0,
            'new_Gaussian_fits_in_this_decision':0,'new_noise_draws':0,'new_trajectory_integrations':0,
            'failed_gates':[k for k,v in gates.items() if not v],
            'protocol_sha256':sha(protocol),'directional_clarification_sha256':sha(clarification),
            'development_assessment_sha256':sha(output/'assessment.json'),'Adam_development_assessment_sha256':sha(adam_file),
            'decision_source_sha256':sha(Path(__file__)),'official_WENDy':False,'main_open_search_adoption':False,
            'status':'ready_for_frozen_fresh_packets_and_source_qualification' if entered else 'not_entered_dependency_gate_failed'}
    destination=output/'independent-entry-decision.json'
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(record),flush=True)
if __name__=='__main__':
    main()
