import numpy as np
import pytest
from wphid.weak.paired_noise import paired_residual_contrast


def test_shared_error_cancels_in_paired_candidate_difference():
    noise=np.arange(12,dtype=float).reshape(6,2)
    result=paired_residual_contrast(first=noise+1,second=noise+3,
        baseline_first=np.ones(2),baseline_second=np.full(2,3.))
    assert result["paired_to_independent_variance_trace_ratio"]==0.
    np.testing.assert_array_equal(result["mean_contrast_shift"],np.zeros(2))


def test_opposite_responses_make_independence_underestimate_variance():
    noise=np.arange(12,dtype=float).reshape(6,2)
    result=paired_residual_contrast(first=noise,second=-noise,
        baseline_first=np.zeros(2),baseline_second=np.zeros(2))
    assert result["paired_to_independent_variance_trace_ratio"]==pytest.approx(2.)


def test_mismatched_draw_counts_rejected():
    with pytest.raises(ValueError):paired_residual_contrast(first=np.zeros((6,2)),second=np.zeros((5,2)),
        baseline_first=np.zeros(2),baseline_second=np.zeros(2))
