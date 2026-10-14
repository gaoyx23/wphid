import json
import numpy as np
from experiments import run_sensor_local_sparse_optimizer_comparison as comparison
from experiments import audit_sensor_transport_stencil_access as access


def test_official_optimizer_rms_scaling_preserves_unthresholded_fit():
    rng=np.random.default_rng(211)
    matrix=rng.normal(size=(100,3))*np.array([1e-3,1,1e3])
    truth=np.array([2.,-.5,.03]);target=matrix@truth
    fitted,support,_=comparison.official_fit(matrix,target,0.)
    np.testing.assert_allclose(fitted,truth,rtol=1e-8,atol=1e-8)
    assert support==[0,1,2]


def test_saved_matched_scope_and_explicit_support_differences():
    data=json.loads((comparison.OUT/"audit.json").read_text())
    assert data["complete"] and data["constrained_selection_reused_not_retrained"]
    assert data["heldout_not_used_for_tuning"]
    assert data["new_simulations"]==data["new_random_realizations"]==0
    differences=[]
    for level,equations in data["records"].items():
        for equation,scales in equations.items():
            for scale,row in scales.items():
                support=row["constrained_saved"]["selected"]["support"]
                assert row["unrestricted_internal"]["selected"]["support"]==support
                if row["official_stlsq"]["selected"]["support"]!=support:
                    differences.append((level,equation,scale))
    assert differences==[("three_tasks","crack_evolution","local_transient")]


def test_pointwise_control_does_not_tune_on_common_weak_holdout():
    data=json.loads((comparison.OUT/"pointwise_control/audit.json").read_text())
    assert data["complete"] and data["no_heldout_tuning"]
    assert not data["native_full_pipeline_claim"]
    assert data["optimizer_selection_calls"]==4
    assert data["new_training_pointwise_blocks"]==6
    assert data["new_heldout_weak_projections"]==12
    for equations in data["records"].values():
        assert equations["crack_evolution"]["official_stlsq"]["selected"]["support"]==["stress_threshold_truth"]


def test_sparse_stencil_access_and_dense_control():
    sparse=access.stencil_access([0,5,15,30],range(8),31,8)
    assert sparse["measured_spatial_cells"]==32
    assert sparse["fully_measured_adjacent_axial_faces"]==0
    assert sparse["directly_accessible_interior_axial_stencils"]==0
    assert sparse["directly_accessible_depth_stencils"]==6
    assert not sparse["global_observability_claim"]
    dense=access.stencil_access(range(31),range(8),31,8)
    assert dense["directly_accessible_interior_axial_stencils"]==29
    assert dense["fully_measured_adjacent_axial_faces"]==30


def test_provenance_corrections_preserve_original_artifacts():
    data=json.loads((comparison.OUT/"provenance_check/audit.json").read_text())
    assert data["complete"] and not data["original_artifacts_modified"]
    assert data["new_fits"]==0 and data["official_source_hash_verified"]
    assert len(data["metadata_corrections"])==2
    assert data["metadata_corrections"][0]["actual_new_heldout_weak_projections"]==12
    for equations in data["pointwise_training_reconstruction_checks"].values():
        assert all(row["byte_identical_reconstructed_training_signature"] for row in equations.values())


def test_full_discrete_operator_control_does_not_claim_recovery():
    root=comparison.ROOT/"artifacts/_audits/full_discrete_transport_operator_control"
    result=json.loads((root/"audit.json").read_text())
    assert result["complete"] and result["new_weak_projections"]==36
    assert result["decision"]["fit_diagnostics"]["attempt_count"]==54
    assert not result["hard_pruning_authorized"]
    assert not result["joint_coverage_calibrated"]
    assert result["decision"]["selected"]=="null_constant_diffusion"
    paired=json.loads((root/"reporting_comparison/audit.json").read_text())
    old,new=paired["records"].values()
    assert set(old["one_se_candidates"])==set(new["one_se_candidates"])
    assert set(old["historical_paired_rule_candidates"])==set(new["historical_paired_rule_candidates"])
    assert paired["new_fits"]==paired["new_projections"]==0
