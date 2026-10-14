import json
import numpy as np
import pytest
from experiments.audit_profiled_transport_nulls import OUTPUT,METHOD


def test_saved_profiled_nulls_preserve_reference_only_scope():
    if not OUTPUT.exists():pytest.skip("saved reference null unavailable")
    report=json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert report["complete"] and not report["independent_confirmation"]
    assert report["new_prefactor_fits"]==report["new_noise_draws"]==0
    for model in report["models"].values():
        assert not model["profile_scope"]["uniform_nuisance_parameter_coverage"]
        assert model["profile_scope"]["reaction_support"]=="oxygen_and_steam"
        scores=model["joint_views"][METHOD]["calibration_scores"]
        assert model["joint_views"][METHOD]["calibrated_squared_radius"]==max(scores)
        eigen=np.asarray(model["joint_views"][METHOD]["absolute_model"]["noise_metric"]["correlation_eigenvalues"])
        assert eigen.min()>-1.e-10
    totals={k:sum(v["summary"][k] for v in report["results"].values()) for k in report["total"]}
    assert totals==report["total"]


def test_completed_shift_packets_match_saved_truth_counts():
    from experiments.validate_profiled_parameter_shift import OUTPUT as SHIFT,CASES
    if not SHIFT.exists():pytest.skip("new coupled shift unavailable")
    report=json.loads(SHIFT.read_text(encoding="utf-8"))
    if not report["complete"]:pytest.skip("new coupled shift still running")
    for case,truth,_,_ in CASES:
        rows=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((SHIFT.parent/"draws"/case).glob("*.json"))]
        assert len(rows)==report["protocol"]["repeats_per_case"]
        assert sum(truth in r["decision"]["candidate_set"] for r in rows)==report["results"][case]["truth_retained"]
        assert sum(not r["decision"]["candidate_set"] for r in rows)==report["results"][case]["empty"]
        for row in rows:assert row["decision"]["parameter_output"] is None
