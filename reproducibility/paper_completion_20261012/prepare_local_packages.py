"""New review archive and verified incremental supplement; no publication."""

import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'docs/paper/manuscript/ress-20261012'
OUTPUT = ROOT / 'artifacts/paper_completion_20261012/release_addendum'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if Path.cwd().resolve() != ROOT or OUTPUT.exists():
        raise ValueError('Wrong workspace or package already prepared; never overwrite')
    audit = read(PAPER / 'assets/review-revision-audit.json')
    if not audit['all_automated_checks_passed']:
        raise ValueError('Review audit must pass before packaging')
    OUTPUT.mkdir(parents=True)
    included, external = {}, {}

    def include(path, expected=None, allow_external=True):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Required file missing/outside workspace: ' + str(path))
        name = path.relative_to(ROOT).as_posix()
        if any(part in {'.git', '.codex', '.agents', '__pycache__'} for part in Path(name).parts):
            raise ValueError('Unrelated/unsafe archive content')
        identity = sha(path)
        if expected is not None and identity != expected:
            raise ValueError('Frozen source/input identity changed: ' + name)
        record = {'sha256': identity, 'bytes': path.stat().st_size}
        if path.suffix == '.npz' and allow_external:
            external[name] = record
        else:
            included[name] = record

    for prefix in ('src/wphid/precision_update_v4', 'src/wphid/observation_moments_v5',
                   'src/wphid/adam_known_support_v2', 'tests/precision_update_v4',
                   'tests/observation_moments_v5', 'tests/adam_known_support_v2',
                   'reproducibility/precision_update_v4', 'reproducibility/observation_moments_v5',
                   'reproducibility/adam_known_support_v2', 'reproducibility/paper_completion_20261012',
                   'artifacts/paper_precision_update_v4', 'artifacts/paper_observation_moments_v5',
                   'artifacts/paper_adam_known_support_v2'):
        for path in (ROOT / prefix).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                include(path)
    for root in ('artifacts/paper_precision_update_v4/conditional_development_20261012',
                 'artifacts/paper_observation_moments_v5/conditional_development_20261012'):
        freeze = read(ROOT / root / 'source-freeze.json')
        inputs = read(ROOT / root / 'input-manifest.json')
        for name, expected in {**freeze['source_sha256'], **inputs['file_sha256']}.items():
            include(ROOT / name, expected)
        seal = read(ROOT / root / 'estimate-seal.json')
        for name, expected in seal['record_sha256'].items():
            include(ROOT / root / name, expected)
    for path in (ROOT / 'docs/research').glob('*20261012-zh.md'):
        include(path)
    base = ROOT / 'artifacts/paper_final_candidate_v1/release_candidate_20261011/wphid-data-code-release-candidate-20261011.zip'
    expected_base = '28d61c3f780a2a043b0d97b93380afc784c21844f907df69c05b702dab6e5670'
    if sha(base) != expected_base:
        raise ValueError('Original V1 base archive identity changed')
    manifest = {'version': 'WPHID_PRECISION_OBSERVATION_DEVELOPMENT_ADDENDUM_20261012',
                'included_files': included, 'required_base_inputs': external,
                'base_archive': base.relative_to(ROOT).as_posix(), 'base_archive_sha256': expected_base,
                'new_scientific_updates': 18, 'new_scientific_fits': 36,
                'new_inputs_or_noise': 0, 'saved_model_replays_completed_before_packaging': 72,
                'V4_and_V5_adoption_gate_passed': False, 'original_eight_package_precision_passes': 6,
                'public_release_completed': False, 'code_license': None,
                'runtime_portability': 'Frozen execution scripts retain exact original runtime locks; this supplement is not a cross-platform execution certification',
                'verification_command': 'python -B reproducibility/paper_completion_20261012/verify_addendum.py'}
    manifest_path = OUTPUT / 'addendum-manifest.json'
    write_new(manifest_path, manifest)
    archive = OUTPUT / 'wphid-precision-observation-development-addendum-20261012.zip'
    with ZipFile(archive, 'x', ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted(included):
            zipped.write(ROOT / name, name)
        zipped.write(manifest_path, manifest_path.relative_to(ROOT).as_posix())
    with ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Supplement archive CRC failure')
        for name, record in included.items():
            if hashlib.sha256(zipped.read(name)).hexdigest() != record['sha256']:
                raise ValueError('Supplement archived byte mismatch: ' + name)
    review_files = {}
    for path in PAPER.rglob('*'):
        if not path.is_file() or any(part in {'exports', '__pycache__'} for part in path.relative_to(PAPER).parts):
            continue
        name = path.relative_to(ROOT).as_posix()
        review_files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    # Preserve repository-relative paths so existing figure links resolve.
    for source in PAPER.glob('*.md'):
        for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', source.read_text(encoding='utf-8')):
            path = (PAPER / target).resolve()
            for actual in (path, path.with_suffix('.png') if path.suffix == '.svg' else path):
                if not actual.is_relative_to(ROOT) or not actual.is_file():
                    raise ValueError('Required review figure missing')
                name = actual.relative_to(ROOT).as_posix()
                review_files[name] = {'sha256': sha(actual), 'bytes': actual.stat().st_size}
    for name, record in included.items():
        if name.startswith('docs/research/') or name.endswith('/assessment.json'):
            review_files[name] = record
    review_manifest = {'files': review_files, 'review_only': True, 'large_input_arrays_included': False,
                       'scientific_gates_passed': False, 'public_or_submission_completed': False}
    review_manifest_path = OUTPUT / 'review-package-manifest.json'
    write_new(review_manifest_path, review_manifest)
    review_archive = PAPER / 'ress-review-package-20261012.zip'
    with ZipFile(review_archive, 'x', ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted(review_files):
            zipped.write(ROOT / name, name)
        zipped.write(review_manifest_path, review_manifest_path.relative_to(ROOT).as_posix())
    with ZipFile(review_archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Review archive CRC failure')
        for name, record in review_files.items():
            if hashlib.sha256(zipped.read(name)).hexdigest() != record['sha256']:
                raise ValueError('Review archived byte mismatch: ' + name)
    identity = {'date': '2026-10-12',
                'addendum': {'file': archive.relative_to(ROOT).as_posix(), 'sha256': sha(archive),
                             'bytes': archive.stat().st_size, 'included_files_verified': len(included)+1,
                             'external_input_arrays': len(external)},
                'review': {'file': review_archive.relative_to(ROOT).as_posix(), 'sha256': sha(review_archive),
                           'bytes': review_archive.stat().st_size, 'files_verified': len(review_files)+1},
                'base_archive_sha256_verified': expected_base,
                'public_release_or_submission_completed': False, 'scientific_confirmation_ready': False,
                'new_scientific_runs_during_packaging': 0}
    write_new(OUTPUT / 'archive-identity.json', identity)
    print(json.dumps(identity))


if __name__ == '__main__':
    main()
