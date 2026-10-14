from dataclasses import replace
import numpy as np
from experiments.validate_profiled_parameter_shift import CASES,SEED,REPEATS,noise_packet,summary
from wphid.truth.reusable_nozzle import reference_nozzle_missions,_hot_fire_duty


def test_shift_cases_are_nonreference_positive_one_factor_changes():
    assert len(CASES)==4 and REPEATS==128
    assert SEED not in (6100819,2026100803,2026100814)
    for _,source,oxygen,steam in CASES:
        assert source in ("Darcy__isotropic","Forchheimer__directional")
        assert oxygen>0 and steam>0
        assert (oxygen!=1)+(steam!=1)==1


def test_shorter_hotfire_has_same_duty_through_observation_window():
    full=reference_nozzle_missions()["high_steam_900s"];short=replace(full,duration=360.)
    for t in np.linspace(0,300,301):assert _hot_fire_duty(t,short)==_hot_fire_duty(t,full)


def test_packet_reproducibility_and_no_hidden_driver_change():
    clean={n:np.ones((3,3,3)) for n in ("T","pO2","pH2O","pProducts","carbon","crack","stress")}
    clean["T"]*=1500;clean["crack"]*=.1
    a=noise_packet(clean,0,2);b=noise_packet(clean,0,2);c=noise_packet(clean,1,2)
    for n in clean:np.testing.assert_array_equal(a[n],b[n])
    for n in ("stress","carbon"):np.testing.assert_array_equal(a[n],clean[n])
    assert not np.array_equal(a["pO2"],c["pO2"])


def test_parameter_error_uses_shifted_truth_only_in_external_summary():
    row=dict(decision=dict(candidate_set=["a"]),profiles=dict(a=dict(amplitude_multipliers=[.8,1.2])))
    result=summary([row],"a",[.8,1.2])
    np.testing.assert_array_equal(result["given_true_structure_prefactor_median_absolute_relative_error"],[0,0])
    assert result["frozen_case_targets_met"]
