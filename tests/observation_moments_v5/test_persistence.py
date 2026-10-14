"""Isolation and immutable-result checks on temporary artificial records."""

import numpy as np
import pytest
from reproducibility.observation_moments_v5 import io


def test_existing_or_outside_record_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'ARTIFACTS', tmp_path / 'v5')
    path = io.ARTIFACTS / 'record.json'
    io.write_new(path, {'artificial': 1})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        io.write_new(path, {'artificial': 2})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        io.write_new(tmp_path / 'escaped.json', {})


def test_numpy_results_are_finite_json_compatible_values():
    assert io.plain({'a': np.array([1., 2.]), 'b': np.bool_(True)}) == {'a': [1., 2.], 'b': True}
