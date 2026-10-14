"""Qualify the new workflow on artificial tests, preserving each attempt."""

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproducibility.precision_update_v4.io import PROTOCOL, ROOT, PREFLIGHT, read, sha, write_new


def main():
    import wphid
    if Path.cwd().resolve() != ROOT or Path(wphid.__file__).resolve() != ROOT / 'src/wphid/__init__.py':
        raise ValueError('Wrong working directory or actual import')
    prior = read(PREFLIGHT / 'source-freeze.json')
    identities = prior['source_sha256']
    for name, expected in identities.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Prior frozen source changed: ' + name)
    if prior['python_version'] != sys.version:
        raise ValueError('Prior Python version differs; qualification required for changed numerical runtime')
    for name, expected in prior['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Prior numerical dependency differs: ' + name)
    parent = ROOT / 'artifacts/paper_precision_update_v4/software_qualification'
    attempt = 1
    while (parent / f'attempt-{attempt:02d}').exists():
        attempt += 1
    output = parent / f'attempt-{attempt:02d}'
    output.mkdir(parents=True)
    files = [*sorted((ROOT / 'src/wphid/precision_update_v4').glob('*.py')),
             *sorted((ROOT / 'tests/precision_update_v4').glob('*.py')),
             *sorted(Path(__file__).resolve().parent.glob('*.py')), Path(__file__).resolve().parent / 'development-protocol.json',
             ROOT / 'reproducibility/paper_completion_20261012/author-instruction.json']
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    env = os.environ.copy()
    env.update({'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT))), 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'})
    command = [sys.executable, '-B', '-m', 'pytest', 'tests/precision_update_v4', '-q',
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
    record = {'version': PROTOCOL['version'], 'tests': len(cases), 'failed': failed, 'skipped': skipped,
              'all_passed': result.returncode == 0 and failed == 0 and skipped == 0,
              'role': 'Artificial software fixtures, not recovery experiments',
              'source_sha256': sources, 'prior_source_sha256': identities,
              'python_version': sys.version, 'python_executable': str(Path(sys.executable).resolve()),
              'prior_python_executable': prior['python_executable'],
              'runtime_path_relocated': prior['python_executable'] != str(Path(sys.executable).resolve()),
              'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pytest')},
              'wphid_import': str(Path(wphid.__file__).resolve()), 'command': command,
              'junit_sha256': sha(output / 'junit.xml'), 'new_scientific_fits': 0}
    write_new(output / 'qualification.json', record)
    print({k: record[k] for k in ('tests', 'failed', 'all_passed', 'runtime_path_relocated')}, flush=True)
    if not record['all_passed']:
        print(result.stdout, flush=True)
        raise SystemExit(result.returncode or 1)


if __name__ == '__main__':
    main()
