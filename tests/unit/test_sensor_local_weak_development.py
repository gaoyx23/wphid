import json
import numpy as np
import pytest
from experiments.run_sensor_local_weak_development import (
    ROOT,OUT,SOURCE,align_raw_channels,digest,estimation_fingerprint)


def test_fixed_replay_raw_hashes_and_truth_not_used_for_pruning():
    result=json.loads((OUT/"audit.json").read_text())
    assert result["complete"]
    assert result["new_physical_trajectories"]==result["new_random_realizations"]==result["transport_fits"]==0
    assert result["fixed_rng_stream_replays"]==2 and result["selection_calls"]==6
    for task,metadata in result["raw_metadata"].items():
        assert all(metadata["all_replayed_fields_bitwise_identical"].values())
        assert len(metadata["all_replayed_fields_bitwise_identical"])==7
        assert metadata["raw_sha256"]==digest(OUT/"raw_channels"/f"{task}.npz")
        assert metadata["old_reconstructed_sha256"]==digest(SOURCE/"observations"/f"{task}.npz")
    for view in result["selection"].values():
        assert not view["reporting"]["hard_pruning_authorized"]
        assert not view["reporting"]["joint_coverage_calibrated"]


def test_scoring_correction_reuses_identical_estimation_code():
    result=json.loads((OUT/"audit.json").read_text())
    assert result["source_correction"]["checkpoint_reused"]
    old=(OUT/"original_runner_source.py").read_text(encoding="utf-8")
    current=(ROOT/"experiments/run_sensor_local_weak_development.py").read_text(encoding="utf-8")
    assert estimation_fingerprint(old)==estimation_fingerprint(current)
    # Actual alignment change must invalidate checkpoint reuse.
    changed=current.replace('np.interp(target_t,times[keep],values[keep,si,xi])',
        'np.interp(target_t,times[keep],values[keep,si,xi])*2')
    assert estimation_fingerprint(old)!=estimation_fingerprint(changed)


def test_alignment_keeps_measured_locations_without_extrapolation():
    raw_path=OUT/"raw_channels/nominal_900s.npz"
    with np.load(raw_path,allow_pickle=False) as raw:
        target_t=raw["times__T"]
    fields,(s_indices,x_indices)=align_raw_channels(raw_path,target_t)
    assert all(v.shape==(len(target_t),4,8) for v in fields.values())
    assert s_indices.tolist()==[0,5,15,30] and len(x_indices)==8
    with pytest.raises(ValueError):align_raw_channels(raw_path,np.array([target_t[0]-1,target_t[-1]+1]))


def test_results_preserve_negative_small_window_and_unresolved_crack():
    result=json.loads((OUT/"audit.json").read_text())
    assert result["scoring_only"]["carbon_evolution"]["preferred_atom_union_exact"]
    assert not result["scoring_only"]["crack_evolution"]["preferred_atom_union_exact"]
    assert result["scoring_only"]["crack_evolution"]["true_support_in_heuristic_candidates"]
    # New processing is not better at every scale: do not hide the trade-off.
    row=result["records"]["lower_load_750s"]["crack_evolution"]["local_transient"]
    assert row["raw_time_aligned"]["relative_residual_to_target"]>row["legacy_at_sensors"]["relative_residual_to_target"]


def test_heldout_coefficients_are_sealed_and_training_unchanged():
    result=json.loads((ROOT/"artifacts/_audits/sensor_local_heldout/audit.json").read_text())
    training=json.loads((OUT/"audit.json").read_text())
    assert result["complete"] and result["new_model_fits"]==0
    assert not result["selection_changed_after_validation"]
    assert result["training_selection_sha256"]==digest(OUT/"audit.json")
    for task,row in result["records"].items():
        assert all(row["raw_metadata"]["all_replayed_fields_bitwise_identical"].values())
        for equation,scales in row["equations"].items():
            for scale,evaluation in scales.items():
                selected=training["selection"][equation]["per_scale"][scale]["selected"]
                assert evaluation["sealed_preferred_support"]==selected["support"]
                block=json.loads((OUT/"blocks/nominal_900s.json").read_text())["equations"][equation][scale]["observed_block"]
                assert evaluation["trained_coefficients"]==[selected["coefficients"][n] for n in block["names"]]
