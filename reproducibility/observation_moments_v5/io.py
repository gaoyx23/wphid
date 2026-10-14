"""Never-overwrite persistence within the independent V5 artifact root."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT))
ARTIFACTS = ROOT / 'artifacts/paper_observation_moments_v5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ARTIFACTS) or path.exists():
        raise ValueError('Existing record or destination outside V5')
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f'.pending-{os.getpid()}')
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError('Concurrent existing record')
    pending.rename(path)


def check_runtime():
    import wphid
    if Path.cwd().resolve() != ROOT or Path(wphid.__file__).resolve() != ROOT / 'src/wphid/__init__.py':
        raise ValueError('Wrong workspace or actual package import')


def plain(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value
