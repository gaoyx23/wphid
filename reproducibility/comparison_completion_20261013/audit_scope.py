"""Read-only evidence audit; never generate inputs or fit an estimator."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
OLD = ROOT / 'artifacts/paper_final_candidate_v1/scientific_confirmation_20261011'
OUTPUT = ROOT / 'artifacts/paper_comparison_completion_20261013'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    import wphid
    if Path.cwd().resolve() != ROOT or not Path(wphid.__file__).resolve().is_relative_to(ROOT / 'src'):
        raise ValueError('Wrong workspace or actual import')
    destination = OUTPUT / 'existing-evidence-audit.json'
    if destination.exists():
        raise FileExistsError('Completed audit already exists; inspect receipt instead')
    frozen = read(OLD / 'source-freeze.json')
    sources = {**frozen['source_sha256'], **frozen['frozen_reference_sha256']}
    for relative, expected in sources.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    manifest = read(OLD / 'input-manifest.json')
    if manifest['source_freeze_sha256'] != sha(OLD / 'source-freeze.json'):
        raise ValueError('Source/input binding mismatch')
    for record in manifest['records']:
        if sha(ROOT / record['file']) != record['sha256']:
            raise ValueError('Input changed: ' + record['file'])
    verification = read(OLD / 'saved-estimate-verification.json')
    # Receipt layout is inspected, rather than accepting directory existence.
    controls = {}
    raw_receipt = json.dumps(verification, sort_keys=True)
    for seed in range(9101101, 9101107):
        for role in ('shared', 'partial', 'independent'):
            path = OLD / 'fits/Q1' / f'{seed}_{role}_wphid.json'
            digest = sha(path)
            if digest not in raw_receipt:
                raise ValueError('Control absent from saved verification: ' + path.name)
            record = read(path)
            if not record['completed'] or record['source_freeze_sha256'] != sha(OLD / 'source-freeze.json'):
                raise ValueError('Unbound or incomplete control')
            controls[path.relative_to(ROOT).as_posix()] = digest
    forward_root = ROOT / 'artifacts/_audits/envelope_unified_stage'
    forward = read(forward_root / 'audit.json')
    status = read(forward_root / 'status.json')
    if not forward['complete'] or status['state'] != 'completed' or sum(status['completed_draws'].values()) != 512:
        raise ValueError('Forward confirmation not complete')
    dependencies = read(ROOT / 'reproducibility/paper_completion_20261012/confirmation-entry-gate.json')
    record = {
        'version': 'COMPARISON_COMPLETION_SCOPE_AUDIT_20261013_V1',
        'actual_cwd': str(ROOT), 'actual_import': str(Path(wphid.__file__).resolve()),
        'python_executable': sys.executable,
        'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'pytest', 'pysindy')},
        'source_freeze_sha256': sha(OLD / 'source-freeze.json'),
        'input_manifest_sha256': sha(OLD / 'input-manifest.json'),
        'source_hashes_verified': sources, 'input_packets_verified': len(manifest['records']),
        'saved_development_control_sha256': controls,
        'forward_audit_sha256': sha(forward_root / 'audit.json'),
        'forward_frozen_model_sha256': forward['frozen_model_sha256'],
        'forward_draws_already_completed': status['completed_draws'],
        'old_new_version_confirmation_entry': dependencies,
        'gitattributes_sha256': sha(ROOT / '.gitattributes'),
        'new_fits': 0, 'new_inputs_or_noise': 0, 'old_science_rerun': False,
        'main_directory_writes': 0, 'original_eight_package_precision_count': '6/8',
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(destination), 'sources': len(sources), 'inputs': len(manifest['records']), 'controls': len(controls)}))


if __name__ == '__main__':
    main()
