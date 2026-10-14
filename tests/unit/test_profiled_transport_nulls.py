import numpy as np
import pytest
from experiments.audit_profiled_transport_nulls import fit_null,summarize,METHOD


def example():
    rng=np.random.default_rng(818)
    shared=rng.normal(size=(128,3))
    samples={"a":shared+.2*rng.normal(size=(128,3)),"b":shared+[2.,1.,-1.]+.2*rng.normal(size=(128,3))}
    ids={n:tuple(range(128)) for n in samples}
    return samples,ids


def test_post_profile_mean_not_forced_to_zero_and_scope_explicit():
    samples,ids=example();model=fit_null("a",samples,ids)
    absolute=model["joint_views"][METHOD]["absolute_model"]
    np.testing.assert_allclose(absolute["noise_metric"]["center"],samples["a"][:64].mean(axis=0))
    assert model["profile_scope"]["null_source_prefactor_multipliers"]==[1.,1.]
    assert not model["profile_scope"]["uniform_nuisance_parameter_coverage"]
    assert not model["profile_scope"]["missing_reaction_support_calibrated"]


def test_evaluation_and_other_sources_never_change_fit_or_threshold():
    samples,ids=example();before=fit_null("a",samples,ids)
    for v in samples.values():v[96:]+=1.e4
    after=fit_null("a",samples,ids)
    assert after["joint_views"][METHOD]["calibrated_squared_radius"]==before["joint_views"][METHOD]["calibrated_squared_radius"]
    np.testing.assert_array_equal(after["contrast_metrics"]["b"]["center"],before["contrast_metrics"]["b"]["center"])


def test_saved_observation_alignment_guard_still_applies():
    samples,ids=example();ids["b"]=tuple(reversed(ids["b"]))
    with pytest.raises(ValueError):fit_null("a",samples,ids)


def test_summary_separates_missed_truth_wrong_unique_and_ambiguity():
    result=summarize("a",[{"candidate_set":["a"]},{"candidate_set":["b"]},{"candidate_set":[]},{"candidate_set":["a","b"]}])
    assert result["truth_retained"]==2
    assert result["unique_correct"]==result["wrong_unique"]==result["empty"]==1
