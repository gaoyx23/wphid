"""Externally drain only already active fits at the user's battery pause request."""
import hashlib
import json
from pathlib import Path
import os
import time
import psutil
import ctypes
from ctypes import wintypes
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'artifacts/paper_comparison_resume_v2/gaussian_fresh_confirmation'
AUDIT=ROOT/'artifacts/paper_comparison_pause_20261014'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def main():
    snapshot=json.loads((OUTPUT/'driver-cpu-checkpoints.jsonl').read_text().splitlines()[-1])
    current=snapshot['states']
    if len(current)!=4:raise ValueError('Exactly the four currently active fits required')
    if len(list((OUTPUT/'fits').glob('*.json')))!=16:raise ValueError('Expected16 complete and4 currently active')
    for state in current:
        process=psutil.Process(state['pid'])
        if abs(process.create_time()-state['process_create_time'])>1e-4:raise ValueError('Active process recycled')
    parents={psutil.Process(s['pid']).ppid() for s in current}
    if len(parents)!=1:raise ValueError('Expected one owned Gaussian driver')
    driver=psutil.Process(next(iter(parents)))
    gate=ROOT/'tests/wendy_comparison_resume_v3/test_orchestration.py'
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    handle=kernel.CreateFileW(str(gate),0x80000000,0,None,3,0x80,None)
    if handle==wintypes.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    driver.suspend()
    save(AUDIT/'current-drain-plan.json',{'user_request':'Finish currently running experiments then pause for battery',
        'already_complete':16,'current_jobs':current,'new_unstarted_scientific_jobs_authorized':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'utc_epoch':time.time(),
        'helper_pid':os.getpid(),'driver_pid_suspended':driver.pid,'exclusive_read_gate_file':str(gate),
        'already_complete_fit_sha256':{p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUTPUT/'fits').glob('*.json')},
        'scientific_source_or_kernel_modified':False,'action':'block next verify reads without altering bytes; suspend each worker after full fit and completion receipt'})
    waiting={s['job_id']:s for s in current};done=[]
    while waiting:
        for identity,state in list(waiting.items()):
            path=OUTPUT/f'fits/{identity}.json'
            if not path.exists():continue
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError,PermissionError):continue
            if not record['completed'] or record['job']['id']!=identity:raise ValueError('Current endpoint incomplete')
            completion=OUTPUT/f'attempt-completion/{identity}.json'
            if not completion.exists():continue
            try:json.loads(completion.read_text(encoding='utf-8'))
            except (json.JSONDecodeError,PermissionError):continue
            process=psutil.Process(state['pid'])
            if abs(process.create_time()-state['process_create_time'])>1e-4:raise ValueError('Active process recycled')
            process.suspend()
            cpu=process.cpu_times()
            observed={'id':identity,'pid':state['pid'],'suspended_after_complete_fit':True,
                'process_cpu_at_suspend':cpu.user+cpu.system,'utc_epoch':time.time(),
                'fit_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'fit_cpu_seconds':record['fit']['cpu_seconds'],
                'attempt_completion_present':(OUTPUT/f'attempt-completion/{identity}.json').exists(),
                'new_start_receipts_after_pause':[p.stem for p in (OUTPUT/'attempt-starts').glob('*.json')
                    if json.loads(p.read_text(encoding='utf-8'))['utc_epoch_at_start']>snapshot['utc_epoch']]}
            save(AUDIT/f'drained-workers/{identity}.json',observed)
            print(json.dumps(observed),flush=True);done.append(observed);del waiting[identity]
        time.sleep(.01)
    save(AUDIT/'current-drain-complete.json',{'all_four_active_fits_complete':True,'selected_complete_fits':20,
        'workers_suspended':done,'remaining_unstarted_cases':34,'scientific_source_modified':False,
        'new_truth_assessment_or_fit_retry':False,'utc_epoch':time.time()})
    print('Current four complete and suspended; external driver shutdown is now safe; gate retained until owned driver/workers stop.',flush=True)
    while psutil.pid_exists(driver.pid) or any(psutil.pid_exists(s['pid']) for s in current):time.sleep(.2)
    kernel.CloseHandle(handle)
    save(AUDIT/'gate-released-after-shutdown.json',{'all_owned_fit_processes_stopped':True,'helper_pid':os.getpid(),'utc_epoch':time.time()})

if __name__=='__main__':main()
