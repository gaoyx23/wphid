import numpy as np
import pytest
from experiments.run_nested_passive_data_pilot import LEVELS,HOLDOUT,TRAIN,validate_split,normalized_error


def test_nested_tasks_and_holdout_isolation():
    validate_split(LEVELS,HOLDOUT)
    assert [len(v) for v in LEVELS.values()]==[2,5,8]
    assert not set(TRAIN)&set(HOLDOUT)


def test_reject_resampled_reordered_or_leaking_levels():
    with pytest.raises(ValueError):validate_split({"a":("x","y"),"b":("y","x","z")},("h",))
    with pytest.raises(ValueError):validate_split({"a":("x","y")},("y",))


def test_validation_keeps_failure_and_checks_training_coefficients():
    assert normalized_error(np.eye(2),np.array([1.,2.]),np.array([1.,2.]))["nrmse"]==0
    assert normalized_error(np.eye(2),np.ones(2),None)==dict(fit_failed=True,nrmse=None)
