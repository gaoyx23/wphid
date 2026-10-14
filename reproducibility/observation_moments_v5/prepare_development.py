"""Lock qualification, old responses and viewed inputs without truth reads."""

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.observation_moments_v5.io import ROOT, ARTIFACTS, read, sha, write_new, check_runtime
from reproducibility.observation_moments_v5.development import OUTPUT, PROTOCOL


def main():
    check_runtime()
    if OUTPUT.exists():
        raise ValueError('Already prepared; never overwrite')
    qualifications = sorted((ARTIFACTS / 'software_qualification').glob('attempt-*/qualification.json'))
    qpath = qualifications[-1]
    q = read(qpath)
    if not q['all_passed']:
        raise ValueError('Latest software qualification failed')
    sources = {**q['old_source_sha256'], **q['source_sha256']}
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source changed since qualification: ' + name)
    prior = ROOT / PROTOCOL['preflight_source_root']
    old = read(prior / 'assessment.json')
    if read(prior / 'execution-complete.json')['assessment_sha256'] != sha(prior / 'assessment.json'):
        raise ValueError('Prior completion identity changed')
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in (prior / 'assessment.json', prior / 'execution-complete.json', qpath,
              ROOT / 'artifacts/paper_final_candidate_v1/scientific_confirmation_20261011/assessment.json',
              ROOT / 'artifacts/paper_precision_update_v4/conditional_development_20261012/assessment.json',
              ROOT / 'reproducibility/paper_completion_20261012/author-instruction.json')}
    points = []
    for seed in PROTOCOL['fixed_seeds']:
        for role in PROTOCOL['roles']:
            path = prior / 'points' / f'{seed}_{role}.json'
            point = read(path)
            if point['status'] != 'computed' or old['record_sha256']['points/' + path.name] != sha(path):
                raise ValueError('Saved input response missing or changed')
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = sha(path)
            files[point['saved_fit']] = point['saved_fit_sha256']
            files[point['input']['file']] = point['input']['sha256']
            points.append({'id': point['id'], 'seed': seed, 'role': role, 'preflight_record': relative})
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen input or old result changed: ' + name)
    freeze = {'version': PROTOCOL['version'], 'frozen_at_UTC': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sources, 'python_version': sys.version,
              'python_executable': str(Path(sys.executable).resolve()), 'versions': q['versions'],
              'qualification_sha256': sha(qpath), 'truth_read_before_estimates': False}
    write_new(OUTPUT / 'source-freeze.json', freeze)
    write_new(OUTPUT / 'input-manifest.json', {'version': PROTOCOL['version'],
              'source_freeze_sha256': sha(OUTPUT / 'source-freeze.json'), 'file_sha256': files,
              'points': points, 'new_inputs_or_noise': 0, 'viewed_development_inputs': True})
    print({'frozen_points': len(points), 'new_fits_if_preflight_passes': PROTOCOL['new_conditional_fits'],
           'output_root': str(OUTPUT)}, flush=True)


if __name__ == '__main__':
    main()
