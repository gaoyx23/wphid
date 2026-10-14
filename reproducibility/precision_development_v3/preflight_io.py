"""Write new preflight records only, with frozen-source and input checks."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
PROTOCOL_PATH = Path(__file__).parent / 'preflight-protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
OUTPUT = ROOT / PROTOCOL['output_root']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT / 'artifacts/paper_precision_development_v3'):
        raise ValueError('Preflight output escaped the independent V3 root')
    if path.exists():
        raise FileExistsError('Existing preflight record must not be overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f'.pending-{os.getpid()}')
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError('Concurrent completed record must not be overwritten')
    pending.rename(path)


def verify_freeze():
    import wphid
    if Path.cwd().resolve() != ROOT or Path(wphid.__file__).resolve() != ROOT / 'src/wphid/__init__.py':
        raise ValueError('Wrong working directory or actual wphid import')
    freeze = read(OUTPUT / 'source-freeze.json')
    if freeze['python_version'] != sys.version or freeze['python_executable'] != str(Path(sys.executable).resolve()):
        raise ValueError('Python runtime differs from freeze')
    for name, expected in freeze['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen preflight/reference source changed: ' + name)
    for name, expected in freeze['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Preflight dependency changed: ' + name)
    manifest = read(OUTPUT / 'input-manifest.json')
    if manifest['source_freeze_sha256'] != sha(OUTPUT / 'source-freeze.json') or len(manifest['points']) != 18:
        raise ValueError('Preflight source/input manifest mismatch')
    for point in manifest['points']:
        for name, expected in ((point['input']['file'], point['input']['sha256']),
                               (point['saved_fit'], point['saved_fit_sha256'])):
            if sha(ROOT / name) != expected:
                raise ValueError('Saved input or fixed point changed: ' + name)
    for name, expected in manifest['prior_metadata_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Prior immutable metadata changed: ' + name)
    return manifest
