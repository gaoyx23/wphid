"""Qualification of V5 on artificial fixtures, separate from science."""

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.observation_moments_v5.io import ROOT, ARTIFACTS, check_runtime, read, sha, write_new


def main():
    check_runtime()
    prior = read(ROOT / 'artifacts/paper_precision_update_v4/conditional_development_20261012/source-freeze.json')
    old = prior['source_sha256']
    for name, expected in old.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen old source changed: ' + name)
    files = [*sorted((ROOT / 'src/wphid/observation_moments_v5').glob('*.py')),
             *sorted((ROOT / 'tests/observation_moments_v5').glob('*.py')),
             *sorted(Path(__file__).resolve().parent.glob('*.py')),
             *sorted(Path(__file__).resolve().parent.glob('*.json')),
             ROOT / 'docs/research/observation-moment-estimator-theory-v5-20261012-zh.md']
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    parent = ARTIFACTS / 'software_qualification'
    attempt = 1
    while (parent / f'attempt-{attempt:02d}').exists():
        attempt += 1
    output = parent / f'attempt-{attempt:02d}'
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update({'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT))), 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
    command = [sys.executable, '-B', '-m', 'pytest', 'tests/observation_moments_v5', '-q',
               '-p', 'no:cacheprovider', '--junitxml', str(output / 'junit.xml')]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
    (output / 'pytest.stdout.log').write_text(result.stdout, encoding='utf-8')
    (output / 'pytest.stderr.log').write_text(result.stderr, encoding='utf-8')
    cases = list(ET.parse(output / 'junit.xml').getroot().iter('testcase'))
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    for name, expected in {**old, **sources}.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source changed during qualification: ' + name)
    record = {'version': 'WPHID_COMMON_OBSERVATION_MOMENTS_V5_20261012', 'tests': len(cases),
              'failed': failed, 'skipped': skipped, 'all_passed': result.returncode == 0 and failed == 0 and skipped == 0,
              'source_sha256': sources, 'old_source_sha256': old,
              'python_version': sys.version, 'python_executable': str(Path(sys.executable).resolve()),
              'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pytest')},
              'command': command, 'junit_sha256': sha(output / 'junit.xml'),
              'artificial_software_fixtures_only': True, 'scientific_precision_or_adoption_established': False}
    write_new(output / 'qualification.json', record)
    print({k: record[k] for k in ('tests', 'failed', 'all_passed')}, flush=True)
    if not record['all_passed']:
        print(result.stdout, flush=True)
        raise SystemExit(result.returncode or 1)


if __name__ == '__main__':
    main()
