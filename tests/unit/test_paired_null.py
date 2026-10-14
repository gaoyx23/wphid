import numpy as np
import pytest
from wphid.weak.empirical_noise import noise_metric
from wphid.weak.paired_null import fit_paired_null, paired_candidate_set


def fixture_models():
    rng=np.random.default_rng(301)
    shared=rng.normal(size=(128,2))
    samples={"a":shared+rng.normal(size=(128,2))*.1,
             "b":shared+[2.,-2.]+rng.normal(size=(128,2))*.1}
    ids={name:tuple(range(128)) for name in samples}
    models={}
    for name in samples:
        absolute=dict(candidate_id=name,candidate_simulated_clean_baseline=[0.,0.],
            noise_metric=noise_metric(center=samples[name][:64].mean(axis=0),covariance=np.eye(2)),
            observation_configuration_id="same")
        models[name]=fit_paired_null(candidate=name,samples=samples,sample_ids=ids,
            absolute_models={"fixed":absolute},observation_configuration_id="same")
    return samples,ids,models


def test_common_residual_shift_is_invisible_to_contrasts_but_not_absolute():
    samples,_,models=fixture_models()
    raw={n:v[96] for n,v in samples.items()}
    shifted={n:v+[100.,-100.] for n,v in raw.items()}
    ids={n:"obs96" for n in raw}
    a=paired_candidate_set(residuals=raw,observation_ids=ids,null_models=models)
    b=paired_candidate_set(residuals=shifted,observation_ids=ids,null_models=models)
    assert a["candidate_set"]==b["candidate_set"]
    for name in raw: assert a["scores"][name]["score"]==pytest.approx(b["scores"][name]["score"])
    guarded=paired_candidate_set(residuals=shifted,observation_ids=ids,null_models=models,absolute_method="fixed")
    assert guarded["candidate_set"]==[]


def test_different_observations_cannot_be_paired():
    samples,_,models=fixture_models()
    with pytest.raises(ValueError,match="one observation"):
        paired_candidate_set(residuals={n:v[96] for n,v in samples.items()},
            observation_ids={"a":"one","b":"two"},null_models=models)


def test_mismatched_training_draw_order_rejected():
    samples,ids,_=fixture_models();ids["b"]=tuple(reversed(ids["b"]))
    with pytest.raises(ValueError,match="identical ordered"):
        fit_paired_null(candidate="a",samples=samples,sample_ids=ids,absolute_models={},observation_configuration_id="same")


def test_evaluation_block_never_changes_fit_or_joint_radius():
    samples,ids,models=fixture_models()
    references={k:v["absolute_model"] for k,v in models["a"]["joint_views"].items()}
    samples["a"][96:]+=100.
    altered=fit_paired_null(candidate="a",samples=samples,sample_ids=ids,
        absolute_models=references,observation_configuration_id="same")
    assert altered["paired_squared_radius"]==models["a"]["paired_squared_radius"]
    assert altered["joint_views"]["fixed"]["calibrated_squared_radius"]==models["a"]["joint_views"]["fixed"]["calibrated_squared_radius"]


def test_missing_candidate_or_view_is_not_silently_ignored():
    samples,_,models=fixture_models()
    with pytest.raises(ValueError):
        paired_candidate_set(residuals={"a":samples["a"][96]},observation_ids={"a":"same"},null_models=models)
