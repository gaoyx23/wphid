"""Preserve interrupted execution identities without starting any estimator."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import psutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper_comparison_completion_20261013'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(folder, stage):
    freeze_path = folder / 'source-freeze.json'
    freeze = read(freeze_path)
    for section in ('source_sha256', 'old_source_sha256', 'old_result_and_input_sha256'):
        for relative, expected in freeze.get(section, {}).items():
            if sha(ROOT / relative) != expected:
                raise ValueError('Frozen identity changed: ' + relative)
    files = folder / 'fits' / stage if stage else folder / 'fits'
    records, partial = {}, {}
    for path in sorted(files.glob('*.json')):
        try:
            record = read(path)
            if not record['completed'] or record['source_freeze_sha256'] != sha(freeze_path):
                raise ValueError('Saved point not bound to frozen source')
            records[path.relative_to(ROOT).as_posix()] = sha(path)
        except (ValueError, KeyError) as error:
            partial[path.relative_to(ROOT).as_posix()] = {'sha256': sha(path), 'reason': str(error)}
    return {'source_freeze_sha256': sha(freeze_path), 'saved_valid_fits': len(records),
            'saved_fit_sha256': records, 'incomplete_or_unbound_files': partial}


def main():
    alive = [p.pid for p in psutil.process_iter(['name']) if p.pid != os.getpid()
             and (p.info['name'] or '').lower().startswith('python')]
    adam = ROOT / 'artifacts/paper_matched_completion_v1'
    gaussian = ROOT / 'artifacts/paper_wendy_comparison_v1/conditional_development_20261013_attempt01'
    result = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Interaction deliberately interrupted; no scientific completion inferred',
        'other_python_processes_observed': alive,
        'adam_development': snapshot(adam, 'development'),
        'fresh_paired_confirmation': snapshot(adam, 'fresh'),
        'gaussian_development': snapshot(gaussian, ''),
        'fresh_declared_jobs': 108, 'gaussian_declared_jobs': 18,
        'adam_development_assessment_sha256': sha(adam / 'development-assessment.json'),
        'fresh_input_manifest_sha256': sha(adam / 'fresh-inputs.json'),
        'uncompleted_inflight_attempt_CPU_and_start_identity': 'not durably checkpointed by frozen runners; audit before continuing, do not silently grant another full allowance',
        'completed_estimates_must_be_reused_by_hash': True,
        'new_fits_performed_by_this_record': 0, 'partial_confirmations_scored_as_complete': False,
        'gaussian_development_truth_scored': (gaussian / 'assessment.json').exists(),
        'fresh_confirmatory_truth_scored': (adam / 'fresh-assessment.json').exists(),
        'whole_task_complete': False, 'main_directory_adoption': False,
        'gitattributes_sha256': sha(ROOT / '.gitattributes')}
    destination = OUTPUT / 'interruption-receipt-20261013.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'alive': alive, 'adam_development': result['adam_development']['saved_valid_fits'],
        'fresh_paired': result['fresh_paired_confirmation']['saved_valid_fits'],
        'gaussian_development': result['gaussian_development']['saved_valid_fits'],
        'task_complete': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
