"""Freeze the authorized 18 updates without evaluating arrays or truth."""

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.precision_update_v4.io import ROOT, PREFLIGHT, OUTPUT, PROTOCOL, read, sha, write_new


def main():
    if Path.cwd().resolve() != ROOT or OUTPUT.exists():
        raise ValueError('Wrong worktree or already prepared; never overwrite')
    authorization = read(ROOT / 'reproducibility/paper_completion_20261012/author-instruction.json')
    if authorization['failed_batch_may_be_extended_until_pass'] or authorization['existing_finished_experiments_may_be_rerun']:
        raise ValueError('Unexpected scope change')
    qualifications = sorted((ROOT / 'artifacts/paper_precision_update_v4/software_qualification').glob('attempt-*/qualification.json'))
    if not qualifications:
        raise ValueError('Software qualification required')
    qualification = qualifications[-1]
    q = read(qualification)
    if not q['all_passed']:
        raise ValueError('Latest qualification failed')
    identities = {**q['prior_source_sha256'], **q['source_sha256']}
    for name, expected in identities.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Qualified source changed: ' + name)
    prior_assessment = read(PREFLIGHT / 'assessment.json')
    if not prior_assessment['practical_screen_passed']:
        raise ValueError('Completed preflight did not qualify this bounded development')
    if read(PREFLIGHT / 'execution-complete.json')['assessment_sha256'] != sha(PREFLIGHT / 'assessment.json'):
        raise ValueError('Prior completion identity mismatch')
    old_manifest = read(PREFLIGHT / 'input-manifest.json')
    files = dict(old_manifest['prior_metadata_sha256'])
    for path in (PREFLIGHT / 'source-freeze.json', PREFLIGHT / 'input-manifest.json',
                 PREFLIGHT / 'assessment.json', PREFLIGHT / 'execution-complete.json', qualification):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    entries = []
    for seed in PROTOCOL['fixed_seeds']:
        for role in PROTOCOL['roles']:
            path = PREFLIGHT / 'points' / f'{seed}_{role}.json'
            point = read(path)
            if point['id'] != f'{seed}_{role}' or point['status'] != 'computed' or point['predicted_steps_applied']:
                raise ValueError('Required saved response missing or already applied')
            for name, expected in ((point['saved_fit'], point['saved_fit_sha256']),
                                   (point['input']['file'], point['input']['sha256'])):
                files[name] = expected
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = sha(path)
            if prior_assessment['record_sha256']['points/' + path.name] != sha(path):
                raise ValueError('Saved response differs from completed screen')
            entries.append({'id': point['id'], 'seed': seed, 'role': role,
                            'preflight_record': relative, 'preflight_sha256': sha(path)})
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Prior input or metadata changed: ' + name)
    freeze = {'version': PROTOCOL['version'], 'frozen_at_UTC': datetime.now(timezone.utc).isoformat(),
              'source_sha256': identities, 'qualification': qualification.relative_to(ROOT).as_posix(),
              'qualification_sha256': sha(qualification), 'python_version': sys.version,
              'python_executable': str(Path(sys.executable).resolve()), 'versions': q['versions'],
              'prior_runtime_path_relocated': q['runtime_path_relocated'],
              'truth_read_before_estimates': False, 'new_nonlinear_optimizations': 0}
    write_new(OUTPUT / 'source-freeze.json', freeze)
    manifest = {'version': PROTOCOL['version'], 'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
                'points': entries, 'file_sha256': files, 'new_inputs_or_noise': 0,
                'source_arrays_evaluated_during_preparation': False, 'selection_uses_truth': False}
    write_new(OUTPUT / 'input-manifest.json', manifest)
    print({'frozen_points': len(entries), 'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json'),
           'output_root': str(OUTPUT)}, flush=True)


if __name__ == '__main__':
    main()
