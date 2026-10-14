"""Freeze a fixed-point diagnostic before scientific input evaluation."""

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys

from preflight_io import OUTPUT, PROTOCOL, ROOT, read, sha, write_new


def main():
    if Path.cwd().resolve() != ROOT or OUTPUT.exists():
        raise ValueError('Wrong worktree or preflight already prepared; never overwrite')
    qualifications = sorted((ROOT / 'artifacts/paper_precision_development_v3/software_qualification_r1').glob('attempt-*/qualification.json'))
    if not qualifications:
        raise ValueError('Software qualification required')
    qualification = qualifications[-1]
    q = read(qualification)
    if not q['all_passed']:
        raise ValueError('Latest software qualification failed')
    identities = {**q['frozen_V1_V2_sha256'], **q['source_sha256']}
    for name, expected in identities.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source changed since qualification: ' + name)
    decisions = ROOT / 'reproducibility/final_candidate_v1/user-decisions.json'
    if not read(decisions)['scientific_execution']['supersedes_earlier_no_new_experiment_instruction']:
        raise PermissionError('Recorded authorization for necessary new scientific work required')
    prior = ROOT / PROTOCOL['prior_root']
    points = []
    for seed in PROTOCOL['fixed_seeds']:
        for role in PROTOCOL['roles']:
            path = prior / f'fits/Q1/{seed}_{role}_wphid.json'
            record = read(path)
            if not record.get('completed') or record['status'] != 'success':
                raise ValueError('Required completed saved point missing')
            source = record['job']['input']
            if sha(ROOT / source['file']) != source['sha256'] or source['seed'] != seed or source['role'] != role:
                raise ValueError('Saved observation identity mismatch')
            points.append({'id': f'{seed}_{role}', 'seed': seed, 'role': role,
                           'input': source, 'saved_fit': path.relative_to(ROOT).as_posix(),
                           'saved_fit_sha256': sha(path)})
    # No targets/states/true parameters are evaluated while freezing this list.
    metadata = [prior / 'source-freeze.json', prior / 'input-manifest.json', prior / 'assessment.json',
                ROOT / PROTOCOL['V2_root'] / 'source-freeze.json',
                ROOT / PROTOCOL['V2_root'] / 'assessment.json', qualification,
                ROOT / 'artifacts/paper_precision_development_v3/moment_preflight_20261011/assessment.json',
                ROOT / 'artifacts/paper_precision_development_v3/moment_preflight_20261011/source-freeze.json']
    freeze = {'version': PROTOCOL['version'], 'frozen_at_UTC': datetime.now(timezone.utc).isoformat(),
              'source_sha256': identities, 'qualification': qualification.relative_to(ROOT).as_posix(),
              'qualification_sha256': sha(qualification),
              'python_executable': str(Path(sys.executable).resolve()), 'python_version': sys.version,
              'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pytest')},
              'truth_read': False, 'new_nonlinear_fits': 0}
    write_new(OUTPUT / 'source-freeze.json', freeze)
    manifest = {'version': PROTOCOL['version'], 'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
                'points': points, 'prior_metadata_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in metadata},
                'new_inputs': 0, 'new_nonlinear_parameter_optimizations': 0,
                'selection_uses_accuracy_or_truth': False, 'scientific_arrays_evaluated': False}
    write_new(OUTPUT / 'input-manifest.json', manifest)
    print({'saved_points': len(points), 'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'),
           'input_manifest_sha256': sha(OUTPUT / 'input-manifest.json')}, flush=True)


if __name__ == '__main__':
    main()
