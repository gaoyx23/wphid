"""Failure-sensitive bookkeeping on artificial fixtures only."""

from types import SimpleNamespace
import numpy as np
import pytest

from reproducibility.precision_update_v4.workflow import digest_array, predict, paired_ratio


def test_common_input_prediction_uses_saved_coefficients_without_refit(monkeypatch):
    import reproducibility.precision_update_v4.workflow as workflow
    matrices = (np.array([[1., 2.], [3., 1.]]),) * 3
    monkeypatch.setattr(workflow, 'design', lambda *args: matrices)
    view = SimpleNamespace(targets=(np.array([5., 5.]),) * 3)
    result = predict(view, None, np.array([.4]), (np.array([1., 2.]),) * 3)
    assert result['equation_rss'] == [0., 0., 0.]
    assert result['prediction_sha256'] == [digest_array([5., 5.])] * 3


def test_invalid_pair_is_not_dropped_to_improve_median():
    rows = [{'id': 'a', 'role': 'partial', 'assessable': True, 'case_score': 1.},
            {'id': 'b', 'role': 'partial', 'assessable': False, 'case_score': None}]
    references = {r['id']: {'assessable': True, 'case_score': 2.} for r in rows}
    assert paired_ratio(rows, references, 'partial') is None


def test_ratio_is_paired_before_aggregation():
    rows = [{'id': 'a', 'role': 'independent', 'assessable': True, 'case_score': 1.},
            {'id': 'b', 'role': 'independent', 'assessable': True, 'case_score': 8.}]
    references = {'a': {'assessable': True, 'case_score': 2.}, 'b': {'assessable': True, 'case_score': 4.}}
    assert paired_ratio(rows, references, 'independent') == 1.25


def test_prediction_digest_preserves_shape_and_float_values():
    assert digest_array(np.ones((2, 2))) != digest_array(np.ones(4))
    assert digest_array([1., 2.]) != digest_array([1., 3.])
