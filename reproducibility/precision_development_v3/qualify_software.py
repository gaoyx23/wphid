"""Preserve each software qualification attempt on artificial fixtures."""

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from preflight_io import ROOT, PROTOCOL_PATH, read, sha, write_new


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Wrong worktree')
    import wphid
    if Path(wphid.__file__).resolve() != ROOT / 'src/wphid/__init__.py':
        raise ValueError('Wrong actual wphid import')
    prior = ROOT / 'artifacts/paper_precision_development_v2/conditional_development_20261011/source-freeze.json'
    identities = read(prior)['source_sha256']
    for name, expected in identities.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen V1/V2 source changed: ' + name)
    parent = ROOT / 'artifacts/paper_precision_development_v3/software_qualification'
    attempt = 1
    while (parent / f'attempt-{attempt:02d}').exists():
        attempt += 1
    output = parent / f'attempt-{attempt:02d}'
    output.mkdir(parents=True)
    files = [*sorted((ROOT / 'src/wphid/precision_development_v3').glob('*.py')),
             *sorted((ROOT / 'tests/precision_development_v3').glob('*.py')),
             *sorted(Path(__file__).parent.glob('*.py')), PROTOCOL_PATH,
             ROOT / 'docs/research/correlated-input-moment-preflight-design-20261011-zh.md']
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'})
    command = [sys.executable, '-B', '-m', 'pytest', 'tests/precision_development_v3', '-q',
               '-p', 'no:cacheprovider', '--junitxml', str(output / 'junit.xml')]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
    (output / 'pytest.stdout.log').write_text(result.stdout, encoding='utf-8')
    (output / 'pytest.stderr.log').write_text(result.stderr, encoding='utf-8')
    cases = list(ET.parse(output / 'junit.xml').getroot().iter('testcase'))
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    for name, expected in {**identities, **sources}.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source changed during qualification: ' + name)
    qualified = result.returncode == 0 and failed == 0 and skipped == 0
    record = {'version': 'WPHID_CORRELATED_INPUT_MOMENT_PREFLIGHT_V3', 'tests': len(cases),
              'failed': failed, 'skipped': skipped, 'all_passed': qualified,
              'role': 'Artificial software fixtures; not mechanism-recovery evidence',
              'source_sha256': sources, 'frozen_V1_V2_sha256': identities,
              'python_executable': str(Path(sys.executable).resolve()), 'python_version': sys.version,
              'wphid_import': str(Path(wphid.__file__).resolve()),
              'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pytest')},
              'command': command, 'junit_sha256': sha(output / 'junit.xml'),
              'stdout_sha256': sha(output / 'pytest.stdout.log'),
              'new_nonlinear_fits': 0, 'new_scientific_noise_draws': 0}
    write_new(output / 'qualification.json', record)
    print({'qualification': output.relative_to(ROOT).as_posix(), 'tests': len(cases),
           'failed': failed, 'all_passed': qualified}, flush=True)
    if not qualified:
        print(result.stdout, flush=True)
        raise SystemExit(result.returncode or 1)


if __name__ == '__main__':
    main()
