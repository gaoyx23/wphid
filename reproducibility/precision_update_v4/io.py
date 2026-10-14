"""Independent V4 persistence; no writes to V1/V2/V3 or main directory."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
PROTOCOL_PATH = Path(__file__).parent/'development-protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
OUTPUT = ROOT/PROTOCOL['output_root']
PREFLIGHT = ROOT/PROTOCOL['preflight_root']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT/'artifacts/paper_precision_update_v4') or path.exists():
        raise ValueError('Output outside isolated V4 or existing record')
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+f'.pending-{os.getpid()}')
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError('Concurrent output already completed')
    pending.rename(path)


def verify():
    import wphid
    if Path.cwd().resolve() != ROOT or Path(wphid.__file__).resolve() != ROOT/'src/wphid/__init__.py':
        raise ValueError('Wrong worktree or actual import')
    freeze = read(OUTPUT/'source-freeze.json')
    for name, expected in freeze['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    if freeze['python_version'] != sys.version or freeze['python_executable'] != str(Path(sys.executable).resolve()):
        raise ValueError('Frozen runtime changed')
    for name, expected in freeze['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Frozen dependency changed')
    manifest = read(OUTPUT/'input-manifest.json')
    for name, expected in manifest['file_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen input or prior result changed: '+name)
    if manifest['source_freeze_sha256'] != sha(OUTPUT/'source-freeze.json'):
        raise ValueError('Manifest source mismatch')
    return manifest
