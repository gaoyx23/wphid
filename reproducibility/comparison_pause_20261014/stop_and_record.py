"""Stop only fully drained owned processes and preserve an explicit pause receipt."""
import hashlib
import json
import os
from pathlib import Path
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper_comparison_resume_v2/gaussian_fresh_confirmation'
AUDIT = ROOT / 'artifacts/paper_comparison_pause_20261014'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Unexpected workspace')
    if (AUDIT / 'paused-state.json').exists():
        raise FileExistsError('Pause already recorded; do not repeat shutdown')
    plan = read(AUDIT / 'current-drain-plan.json')
    drained = read(AUDIT / 'current-drain-complete.json')
    if not drained['all_four_active_fits_complete']:
        raise ValueError('Do not stop active estimates')
    expected_ids = set(plan['already_complete_fit_sha256']) | {s['job_id'] for s in plan['current_jobs']}
    paths = list((OUTPUT / 'fits').glob('*.json'))
    if len(paths) != 20 or {p.stem for p in paths} != expected_ids:
        raise ValueError('Expected original16 plus exactly four drained fits')
    for identifier, expected in plan['already_complete_fit_sha256'].items():
        if sha(OUTPUT / 'fits' / (identifier + '.json')) != expected:
            raise ValueError('Previously saved point changed')
    for record in drained['workers_suspended']:
        if sha(OUTPUT / 'fits' / (record['id'] + '.json')) != record['fit_sha256']:
            raise ValueError('Drained complete point changed')
    frozen = read(OUTPUT / 'source-freeze.json')
    jobs = {j['id']: j for j in frozen['jobs']}
    for path in paths:
        record = read(path)
        if not record['completed'] or record['job'] != jobs[path.stem] or record['source_freeze_sha256'] != sha(OUTPUT / 'source-freeze.json'):
            raise ValueError('Saved endpoint identity differs')
    starts = list((OUTPUT / 'attempt-starts').glob('*.json'))
    if {p.stem for p in starts} != expected_ids:
        raise ValueError('Unexpected additional scientific attempt; audit before shutdown')
    driver = psutil.Process(plan['driver_pid_suspended'])
    command = driver.cmdline()
    if not any('wendy_comparison_resume_v3' in argument and 'complete.py' in argument for argument in command) or command[-1] != 'run':
        raise ValueError('Not the owned experiment driver')
    processes = []
    for state in plan['current_jobs']:
        process = psutil.Process(state['pid'])
        if abs(process.create_time() - state['process_create_time']) > 1e-4 or process.ppid() != driver.pid:
            raise ValueError('Worker process ownership changed')
        processes.append(process)
    stopped = []
    for process in processes + [driver]:
        process.terminate()
        process.wait(timeout=10)
        stopped.append(process.pid)
    deadline = time.monotonic() + 10
    while not (AUDIT / 'gate-released-after-shutdown.json').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Pause helper has not released its read gate')
        time.sleep(.1)
    for section in ('source_sha256', 'receipt_sha256'):
        for relative, expected in frozen[section].items():
            if sha(ROOT / relative) != expected:
                raise ValueError('Frozen source or prerequisite changed during drain')
    for job in frozen['jobs']:
        if sha(ROOT / job['input_file']) != job['input_sha256']:
            raise ValueError('Frozen observation changed')
    if any(psutil.pid_exists(pid) for pid in stopped):
        raise ValueError('An owned experiment process remains')
    fit_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    receipts = [AUDIT / n for n in ('current-drain-plan.json', 'current-drain-complete.json', 'gate-released-after-shutdown.json')]
    save(AUDIT / 'paused-state.json', {
        'status': 'paused_at_explicit_user_request_after_current_four_finished',
        'user_reason': 'Computer battery is running low',
        'Gaussian_fresh_completed': 20, 'Gaussian_fresh_declared': 54, 'Gaussian_fresh_remaining': 34,
        'paired_confirmation_complete': 108, 'Adam_development_complete': 18, 'Gaussian_development_complete': 18,
        'fit_sha256': fit_hashes,
        'remaining_jobs': [j for j in frozen['jobs'] if j['id'] not in expected_ids],
        'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
        'pause_receipt_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in receipts},
        'shutdown_source_sha256': sha(Path(__file__)),
        'owned_driver_and_worker_pids_stopped': stopped,
        'current_four_drained_without_lost_fit_state': True,
        'new_scientific_attempts_after_pause_request': 0,
        'partial_batch_truth_assessment': False,
        'new_noise_draws_or_integrations': 0,
        'frozen_sources_and_inputs_unchanged': True,
        'do_not_auto_resume': True,
        'resume_only_remaining34_reuse_completed20': True,
        'final_method_comparison_complete': False,
        'recorded_utc_epoch': time.time(),
    })
    print(json.dumps({'paused': True, 'Gaussian_saved': 20, 'remaining': 34, 'stopped_pids': stopped}))


if __name__ == '__main__':
    main()
