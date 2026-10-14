"""Read frozen old inputs; write only the new development result tree."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
PROTOCOL_PATH=Path(__file__).parent/'development-protocol.json'
PROTOCOL=json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
OUTPUT=ROOT/PROTOCOL['output_root']
PRIOR=ROOT/PROTOCOL['prior_confirmation_root']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT/'artifacts/paper_precision_development_v2'):
        raise ValueError('Development output escaped isolated V2 root')
    if path.exists():raise FileExistsError('Refuse overwrite of saved development record: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+f'.pending-{os.getpid()}')
    with pending.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if path.exists():raise FileExistsError('Concurrent completed record must not be overwritten')
    pending.rename(path)


def verify_freeze():
    import wphid
    if Path.cwd().resolve()!=ROOT or Path(wphid.__file__).resolve()!=ROOT/'src/wphid/__init__.py':
        raise ValueError('Wrong working directory or wphid import')
    freeze=read(OUTPUT/'source-freeze.json')
    if freeze['python_version']!=sys.version or freeze['python_executable']!=str(Path(sys.executable).resolve()):
        raise ValueError('Python runtime differs from development freeze')
    for name,expected in freeze['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen development/reference source changed: '+name)
    for name,expected in freeze['versions'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError('Development runtime changed: '+name)
    manifest=read(OUTPUT/'input-manifest.json')
    if manifest['source_freeze_sha256']!=sha(OUTPUT/'source-freeze.json'):
        raise ValueError('Input manifest/source freeze mismatch')
    if (len(manifest['jobs'])!=PROTOCOL['new_estimates']
            or len(manifest['unchanged_controls'])!=PROTOCOL['reused_unchanged_shared_controls']
            or any(j['role'] not in PROTOCOL['new_fit_roles'] for j in manifest['jobs'])):
        raise ValueError('Manifest differs from fixed changed-path/control counts')
    for item in manifest['jobs']+manifest['unchanged_controls']:
        for name,expected in [(item['input']['file'],item['input']['sha256']),
                              (item['V1_fit'],item['V1_fit_sha256'])]:
            if sha(ROOT/name)!=expected:raise ValueError('Prior observation/comparator changed: '+name)
    for name,expected in manifest['prior_metadata_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Prior metadata changed: '+name)
    return manifest
