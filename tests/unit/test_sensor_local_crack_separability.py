import json
import hashlib
import numpy as np
from experiments.audit_sensor_local_crack_separability import ROOT,OUT,residualize,diagnose,PRIMARY,SECONDARY,ALTERNATIVE


def test_residual_geometry_invariant_to_basis_units_and_duplicates():
    values=np.array([1.,2.,4.,8.])
    basis=np.array([1.,2.,1.,0.])
    expected=residualize(values,basis)
    np.testing.assert_allclose(residualize(values,np.column_stack((basis*1e-9,basis*1e9))),expected,atol=1e-14)
    assert abs(expected@basis)<1e-13


def test_competitor_can_absorb_entire_secondary_without_claiming_absence():
    primary=np.array([1.,0.,0.]);secondary=np.array([0.,1.,0.])
    matrix=np.column_stack((primary,secondary,2*secondary))
    result=diagnose(matrix,primary+secondary,[PRIMARY,SECONDARY,ALTERNATIVE],1,1)
    geometry=result["geometry_truth_blind"]
    assert geometry["secondary_fraction_outside_primary_and_alternative"]<1e-14
    assert not geometry["degenerate_secondary_column"]
    assert "no automatic threshold" in result["interpretation"]


def test_audit_reads_saved_scores_without_fits_or_selection_changes():
    result=json.loads((OUT/"audit.json").read_text(encoding="utf-8"))
    assert result["complete"] and result["original_selection_unchanged"]
    assert result["new_model_fits"]==result["new_simulations"]==result["new_random_realizations"]==result["new_weak_projections"]==0
    for relative,digest in result["input_hashes"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    for row in result["records"].values():
        assert row["score_differences"]["alternative_minus_chemistry"]>0
        assert row["score_differences"]["simple_minus_chemistry"]<row["score_differences"]["best_model_one_se_width"]


def test_one_cycle_extension_reuses_old_blocks_and_remains_development():
    directory=ROOT/"artifacts/_audits/sensor_local_natural_cycle_extension"
    result=json.loads((directory/"audit.json").read_text(encoding="utf-8"))
    assert result["complete"] and result["adaptive_development"]
    assert result["new_simulations"]==result["new_random_realizations"]==result["transport_fits"]==0
    assert result["fixed_rng_stream_replays"]==1
    assert result["new_sensor_weak_blocks"]==result["new_selection_calls"]==6
    assert result["added_task"]=="reflight_ground_cycle3_900s"
    for task,digest in result["reused_training_block_hashes"].items():
        path=ROOT/"artifacts/_audits/sensor_local_weak_development/blocks"/f"{task}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    for view in result["selection"].values():
        assert not view["reporting"]["hard_pruning_authorized"]
        assert not view["reporting"]["joint_coverage_calibrated"]
    crack=result["selection"]["crack_evolution"]["per_scale"]
    for scale in crack.values():
        assert len(scale["selected"]["one_standard_error_supports"])==2
        assert all(len(s)==2 for s in scale["selected"]["one_standard_error_supports"])
    assert crack["local_transient"]["selected"]["support"]!=crack["global"]["selected"]["support"]
