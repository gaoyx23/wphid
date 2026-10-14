import numpy as np
from reproducibility.matched_completion_resume_v2.replay_r1 import fitting_rss

def test_fitting_rss_applies_target_weights_not_physical_rss():
    y=(np.asarray([3.,5.]),np.asarray([2.,4.]))
    p=(np.asarray([1.,4.]),np.asarray([1.,2.]))
    weights=(.5,3.)
    actual=fitting_rss(y,p,weights)
    np.testing.assert_array_equal(actual,np.asarray([1.25,45.]))
    np.testing.assert_allclose(actual,[np.sum(((a-b)*w)**2) for a,b,w in zip(y,p,weights)],rtol=0,atol=0)

def test_changed_physical_units_preserve_fitting_rss():
    y=(np.asarray([3.,5.]),);p=(np.asarray([1.,4.]),);weight=(.5,)
    np.testing.assert_allclose(fitting_rss(y,p,weight),fitting_rss(tuple(a*1000 for a in y),tuple(a*1000 for a in p),(.0005,)),rtol=0,atol=0)
