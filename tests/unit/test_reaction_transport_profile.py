import numpy as np
import pytest
from wphid.weak.empirical_noise import noise_metric
from wphid.weak.reaction_transport_profile import fit_reaction_profile,project_reaction_source


def problem():
    columns=np.array([[1.,0.],[0.,2.],[1.,2.],[2.,-1.]])
    amplitudes=np.array([.7,1.4])
    residual=columns@(amplitudes-1)
    metric=noise_metric(center=np.ones(4)*999,covariance=np.eye(4))
    return residual,columns,metric,amplitudes


def test_joint_prefactors_recovered_without_metric_center_or_truth_label():
    residual,columns,metric,amplitudes=problem()
    result=fit_reaction_profile(full_reaction_residual=residual,omission_columns=columns,metric=metric,support="oxygen_and_steam")
    np.testing.assert_allclose(result["amplitude_multipliers"],amplitudes)
    assert result["weighted_squared_residual"]<1.e-25
    assert result["normalized_design_rank"]==2
    assert not result["calibrated_support_selection"]


def test_units_do_not_change_fitted_parameters():
    residual,columns,metric,_=problem();units=np.array([1.e-6,1.e6,2.,3.])
    reference=fit_reaction_profile(full_reaction_residual=residual,omission_columns=columns,metric=metric,support="oxygen_and_steam")
    changed=fit_reaction_profile(full_reaction_residual=residual*units,omission_columns=columns*units[:,None],
        metric=noise_metric(center=np.zeros(4),covariance=np.diag(units**2)),support="oxygen_and_steam")
    np.testing.assert_allclose(changed["amplitude_multipliers"],reference["amplitude_multipliers"])


def test_omitted_support_is_not_fitted_back_in():
    residual,columns,metric,_=problem()
    result=fit_reaction_profile(full_reaction_residual=residual,omission_columns=columns,metric=metric,support="oxygen_only")
    assert result["amplitude_multipliers"][1]==0
    assert result["weighted_squared_residual"]>0


def test_negative_unconstrained_prefactor_is_blocked():
    columns=np.eye(2);metric=noise_metric(center=np.zeros(2),covariance=np.eye(2))
    result=fit_reaction_profile(full_reaction_residual=np.array([-2.,0.]),omission_columns=columns,metric=metric,support="oxygen_and_steam")
    np.testing.assert_array_equal(result["amplitude_multipliers"],[0.,1.])
    assert result["fitted_active_coefficients_at_zero"]==[0]


def test_collinear_design_does_not_claim_identifiable_parameters():
    columns=np.ones((4,2));metric=noise_metric(center=np.zeros(4),covariance=np.eye(4))
    result=fit_reaction_profile(full_reaction_residual=np.zeros(4),omission_columns=columns,metric=metric,support="oxygen_and_steam")
    assert result["normalized_design_rank"]==1
    assert result["normalized_design_condition"] is None


def test_source_projection_is_linear_and_rejects_bad_geometry():
    t=np.linspace(0,1,5);s=np.linspace(0,1,5);x=np.linspace(0,1,5)
    args=dict(t=t,s=s,x=x,volume=np.ones((5,5)),window_shape=(5,5,5),stride_shape=(4,4,4))
    source=np.ones((5,5,5))
    a=project_reaction_source(forcing=source,**args)
    np.testing.assert_allclose(project_reaction_source(forcing=-2*source,**args),-2*a)
    with pytest.raises(ValueError):project_reaction_source(forcing=source,**dict(args,volume=-args["volume"]))
