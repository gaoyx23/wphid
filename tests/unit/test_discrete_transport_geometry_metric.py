import numpy as np
import pytest

from experiments.audit_discrete_transport_manufactured import manufactured_inputs
from wphid.weak.discrete_transport import project_discrete_transport_candidate


def test_explicit_unit_metric_preserves_original_operator():
    kwargs=manufactured_inputs(9,"Forchheimer",3.)
    original=project_discrete_transport_candidate(**kwargs)
    kwargs["geometry"]["meridional_face_metric"]=np.ones((8,9))
    explicit=project_discrete_transport_candidate(**kwargs)
    np.testing.assert_array_equal(original["matrix"],explicit["matrix"])
    np.testing.assert_array_equal(original["target"],explicit["target"])
    assert original["geometry_metric_mode"]=="unit_reference_metric"
    assert explicit["geometry_metric_mode"]=="declared_meridional_metric"


@pytest.mark.parametrize("metric",[np.zeros((8,9)),np.full((8,9),np.nan),np.ones((9,9))])
def test_bad_metric_is_rejected(metric):
    kwargs=manufactured_inputs(9,"Darcy",1.)
    kwargs["geometry"]["meridional_face_metric"]=metric
    with pytest.raises(ValueError,match="metric"):
        project_discrete_transport_candidate(**kwargs)


def test_metric_changes_meridional_column_without_changing_source_or_thickness_column():
    kwargs=manufactured_inputs(9,"Darcy",3.)
    original=project_discrete_transport_candidate(**kwargs)
    kwargs["geometry"]["meridional_face_metric"]=np.full((8,9),2.)
    explicit=project_discrete_transport_candidate(**kwargs)
    np.testing.assert_allclose(explicit["matrix"][:,0],.5*original["matrix"][:,0],rtol=1.e-14)
    np.testing.assert_array_equal(explicit["matrix"][:,1],original["matrix"][:,1])
    np.testing.assert_array_equal(explicit["target"],original["target"])
