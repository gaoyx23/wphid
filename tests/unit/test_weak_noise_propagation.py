import numpy as np
import pytest

from wphid.weak.uncertainty import linearize_balance_noise


def test_linear_correlated_modes_match_exact_covariance():
    a=np.array([[1.,2.],[-1.,3.],[2.,-2.]])
    modes=np.array([[.2,.1],[.1,-.3]])
    result=linearize_balance_noise(values=np.array([1.,2.]),noise_modes=modes,
        evaluate=lambda x:a@x,mode_labels=("shared","contrast"))
    np.testing.assert_allclose(result["noise_response_factor"],a@modes,atol=2.e-11)
    np.testing.assert_allclose(result["linearized_covariance"],a@modes@modes.T@a.T,atol=2.e-11)
    assert result["effective_noise_rank"]==2
    assert result["residual_outside_noise_range_norm"]<1.e-10
    assert np.max(result["unit_mode_curvature_relative_to_linear_response"])<1.e-10


def test_overlapping_residuals_are_not_treated_as_independent():
    result=linearize_balance_noise(values=np.array([2.]),noise_modes=np.array([[.5]]),
        evaluate=lambda x:np.array([x[0],x[0],1.]),mode_labels=("shared",))
    assert result["linearized_covariance"][0,1]==pytest.approx(.25)
    assert result["effective_noise_rank"]==1
    assert result["residual_outside_noise_range_norm"]==pytest.approx(1.)
    assert result["noise_range_mahalanobis_squared"]==pytest.approx(16.)


def test_nonlinear_curvature_is_reported_despite_stable_local_derivative():
    result=linearize_balance_noise(values=np.array([2.]),noise_modes=np.array([[.4]]),
        evaluate=lambda x:x*x,mode_labels=("scale",))
    assert result["noise_response_factor"][0,0]==pytest.approx(1.6)
    assert result["unit_mode_curvature_relative_to_linear_response"][0]==pytest.approx(.1)
    assert result["derivative_step_relative_change"][0]<1.e-8
    assert not result["calibrated_selection"]


def test_noise_scaling_changes_covariance_quadratically():
    args=dict(values=np.array([2.]),evaluate=lambda x:x,mode_labels=("scale",))
    first=linearize_balance_noise(noise_modes=np.array([[.5]]),**args)
    second=linearize_balance_noise(noise_modes=np.array([[1.]]),**args)
    np.testing.assert_allclose(second["linearized_covariance"],4*first["linearized_covariance"],rtol=1.e-10)


@pytest.mark.parametrize("changes",[
    dict(noise_modes=np.array([[np.nan]])),dict(noise_modes=np.zeros((2,1))),
    dict(mode_labels=("",)),dict(epsilon=0),dict(rank_relative_tolerance=0),
    dict(evaluate=lambda x:np.array([np.nan])),
])
def test_invalid_noise_contracts_rejected(changes):
    args=dict(values=np.array([2.]),noise_modes=np.array([[.5]]),evaluate=lambda x:x,mode_labels=("scale",))
    args.update(changes)
    with pytest.raises(ValueError):
        linearize_balance_noise(**args)


def test_zero_response_rank_does_not_hide_unexplained_residual():
    result=linearize_balance_noise(values=np.array([2.]),noise_modes=np.array([[0.]]),
        evaluate=lambda x:np.array([3.,4.]),mode_labels=("zero",))
    assert result["effective_noise_rank"]==0
    assert result["noise_range_mahalanobis_squared"]==0.
    assert result["residual_outside_noise_range_norm"]==5.


def test_dimension_changes_under_perturbation_are_rejected():
    with pytest.raises(ValueError,match="dimensions changed"):
        linearize_balance_noise(values=np.array([2.]),noise_modes=np.array([[.5]]),
            evaluate=lambda x:np.ones(2 if x[0]>2 else 1),mode_labels=("scale",))
