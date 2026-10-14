import numpy as np
import pytest
from wphid.weak.conditional_profile import fit_conditional_null,conditional_candidate_set


def example():
    rng=np.random.default_rng(381)
    features=rng.normal(size=(128,2))
    noise=rng.normal(size=(128,4))*.2
    samples={"a":noise,"b":noise+features@rng.normal(size=(2,4))+rng.normal(size=(128,4))*.1}
    ids={n:tuple(range(128)) for n in samples}
    args=dict(candidate="a",samples=samples,features=features,sample_ids=ids,
        fit_indices=np.arange(64),calibration_indices_by_anchor={"reference":np.arange(64,96)},configuration_id="same")
    return args


def test_affine_conditioning_and_calibration_are_fitted_without_evaluation():
    args=example();before=fit_conditional_null(**args)
    for v in args["samples"].values():v[96:]+=1000
    args["features"][96:]+=1000
    after=fit_conditional_null(**args)
    assert after["squared_radius"]==before["squared_radius"]
    np.testing.assert_array_equal(after["views"]["b"]["coefficients"],before["views"]["b"]["coefficients"])
    assert not before["unknown_continuous_parameter_coverage_established"]


def test_feature_units_do_not_change_predictions():
    args=example();before=fit_conditional_null(**args)
    args["features"]*=np.array([1.e-6,1.e6]);after=fit_conditional_null(**args)
    np.testing.assert_allclose(before["views"]["b"]["coefficients"],after["views"]["b"]["coefficients"],atol=1.e-12)
    assert after["squared_radius"]==pytest.approx(before["squared_radius"])


def test_overlapping_blocks_and_observation_order_rejected():
    args=example();args["calibration_indices_by_anchor"]={"bad":np.arange(63,95)}
    with pytest.raises(ValueError):fit_conditional_null(**args)
    args=example();args["sample_ids"]["b"]=tuple(reversed(range(128)))
    with pytest.raises(ValueError):fit_conditional_null(**args)


def test_candidate_owned_features_used_without_truth_label():
    args=example();a=fit_conditional_null(**args)
    b=fit_conditional_null(**dict(args,candidate="b"))
    profiles={n:dict(amplitude_multipliers=args["features"][100],profiled_residual=v[100]) for n,v in args["samples"].items()}
    result=conditional_candidate_set(profiles=profiles,observation_ids=dict(a="100",b="100"),null_models=dict(a=a,b=b))
    assert "a" in result["candidate_set"]
    assert result["parameter_output"] is None
    with pytest.raises(ValueError):conditional_candidate_set(profiles=profiles,observation_ids=dict(a="100",b="101"),null_models=dict(a=a,b=b))
