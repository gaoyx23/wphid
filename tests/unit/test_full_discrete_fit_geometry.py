import numpy as np
from experiments.audit_full_discrete_fit_geometry import diagnose


def test_known_cancellation_is_not_hidden_by_small_time_denominator():
    a=np.array([[100.,-99.],[200.,-198.],[300.,-297.]])
    b=np.array([1.,2.,3.]);c=np.ones(2)
    report=diagnose(a,b,c,1)
    assert report["diffusion_reaction_cosine"]<-.999999
    np.testing.assert_allclose(report["diffusion_norm_relative_to_target"],100.,rtol=1e-14)
    np.testing.assert_allclose(report["reaction_norm_relative_to_target"],99.,rtol=1e-14)
    assert report["residual_relative_to_target"]==0
    assert report["smallest_to_largest_singular_ratio"]<1e-12
    assert report["parameter_confidence_interval"] is None


def test_geometry_ratio_does_not_depend_on_dictionary_units():
    rng=np.random.default_rng(162);a=rng.normal(size=(50,3));b=rng.normal(size=50)
    c=np.array([1.,2.,3.]);factors=np.array([1e-4,1,1e4])
    first=diagnose(a,b,c,2);second=diagnose(a*factors,b,c/factors,2)
    np.testing.assert_allclose(first["normalized_singular_values"],second["normalized_singular_values"])
    np.testing.assert_allclose(first["standardized_parameter_effects"],second["standardized_parameter_effects"])
