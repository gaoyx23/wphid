"""Read-only check of extracted supplement bytes and required base inputs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'artifacts/paper_completion_20261012/release_addendum/addendum-manifest.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    missing = []
    for section in ('included_files', 'required_base_inputs'):
        for name, record in manifest[section].items():
            path = ROOT / name
            if not path.exists():
                missing.append(name)
            elif sha(path) != record['sha256']:
                raise ValueError('Hash mismatch: ' + name)
    print(json.dumps({'included_files': len(manifest['included_files']),
                      'required_base_inputs': len(manifest['required_base_inputs']),
                      'missing': missing, 'new_fits_or_replays': 0,
                      'public_or_portable_runtime_certified': False}))
    if missing:
        raise SystemExit(1)
