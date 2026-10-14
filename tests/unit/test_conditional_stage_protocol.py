import numpy as np
from experiments.build_conditional_profile_model import NEW_CASES,datasets
from experiments.run_conditional_profile_stage import CONFIRM_CASES,DEVELOPMENT_SEED,CONFIRMATION_SEED,REPEATS,noise
from experiments.validate_profiled_parameter_shift import CASES as OLD_CASES
from experiments.confirm_fixed_paired_noise_holdout import NAMES


def test_only_missing_parameter_anchors_are_generated():
    assert {c[0] for c in NEW_CASES}.isdisjoint({c[0] for c in OLD_CASES})
    for source in NAMES:
        all_cases=[c for c in (*NEW_CASES,*OLD_CASES) if c[1]==source]
        assert {(c[2],c[3]) for c in all_cases}=={(.8,1.),(1.,1.2)}
        assert len(datasets()[source])==3


def test_confirmation_parameters_are_new_joint_interior_not_anchor_repetition():
    assert len(CONFIRM_CASES)==4 and {c[1] for c in CONFIRM_CASES}==set(NAMES)
    for _,_,oxygen,steam in CONFIRM_CASES:
        assert (oxygen,steam) not in ((1.,1.),(.8,1.),(1.,1.2))
        assert 0<(1-oxygen)/.2+(steam-1)/.2<1
    assert REPEATS==128
    assert DEVELOPMENT_SEED!=CONFIRMATION_SEED
    assert DEVELOPMENT_SEED not in (6100819,2026100803,2026100814,2026100827)


def test_development_and_confirmation_noise_disjoint_and_reproducible():
    fields={n:np.ones((3,3,3)) for n in ("T","pO2","pH2O","pProducts","carbon","crack","stress")}
    fields["T"]*=1500;fields["crack"]*=.1
    a=noise(fields,0,0,DEVELOPMENT_SEED);b=noise(fields,0,0,DEVELOPMENT_SEED);c=noise(fields,0,0,CONFIRMATION_SEED)
    for n in fields:np.testing.assert_array_equal(a[n],b[n])
    assert not np.array_equal(a["pO2"],c["pO2"])
    for n in ("carbon","stress"):np.testing.assert_array_equal(a[n],fields[n])
