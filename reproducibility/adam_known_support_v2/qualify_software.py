"""Preserve artificial comparator qualification attempts and prior hashes."""

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT))
from reproducibility.observation_moments_v5.io import read, sha


def main():
    import wphid
    if Path.cwd().resolve() != ROOT or Path(wphid.__file__).resolve() != ROOT / 'src/wphid/__init__.py':
        raise ValueError('Wrong worktree or actual import')
    old = read(ROOT / 'artifacts/paper_observation_moments_v5/conditional_development_20261012/source-freeze.json')['source_sha256']
    for name, expected in old.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    parent = ROOT / 'artifacts/paper_adam_known_support_v2/software_qualification'
    attempt = 1
    while (parent / f'attempt-{attempt:02d}').exists():
        attempt += 1
    output = parent / f'attempt-{attempt:02d}'
    output.mkdir(parents=True)
    files = [*sorted((ROOT / 'src/wphid/adam_known_support_v2').glob('*.py')),
             *sorted((ROOT / 'tests/adam_known_support_v2').glob('*.py')), Path(__file__).resolve(),
             ROOT / 'reproducibility/final_candidate_v1/official-source-lock.json']
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    env = os.environ.copy()
    env.update({'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT))), 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
    command = [sys.executable, '-B', '-m', 'pytest', 'tests/adam_known_support_v2', '-q',
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
    record = {'version': 'ADAM_SINDY_CONDITIONAL_KNOWN_SUPPORT_V2_20261012', 'tests': len(cases),
              'failed': failed, 'skipped': skipped, 'all_passed': result.returncode == 0 and failed == 0 and skipped == 0,
              'source_sha256': sources, 'prior_source_sha256': old, 'command': command,
              'python_version': sys.version, 'python_executable': str(Path(sys.executable).resolve()),
              'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'pytest')},
              'junit_sha256': sha(output / 'junit.xml'), 'new_scientific_observations_or_fits': 0,
              'artificial_fixture_fits_are_not_paper_recovery_evidence': True,
              'native_official_example_reproduced': False, 'matched_scientific_performance_established': False}
    path = output / 'qualification.json'
    with path.open('x', encoding='utf-8') as stream:
        import json
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print({k: record[k] for k in ('tests', 'failed', 'all_passed')}, flush=True)
    if not record['all_passed']:
        print(result.stdout, flush=True)
        raise SystemExit(result.returncode or 1)


if __name__ == '__main__':
    main()
