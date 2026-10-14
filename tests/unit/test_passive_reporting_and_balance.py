import hashlib
import json
import numpy as np
import pytest
from wphid.discovery.passive_reporting import summarize_passive_supports
from experiments.audit_nested_passive_observation_balance import ROOT, balance_summary, known_coefficients


def test_support_compression_is_explicit_not_a_joint_model_or_safe_pruning():
    evidence={"s1":{"selected":{"support":["a"],"one_standard_error_supports":[["a"],["a","b"]]}},
        "s2":{"selected":{"support":[],"one_standard_error_supports":[[]]}}}
    report=summarize_passive_supports(per_scale=evidence,candidate_names=["a","b"],
        allowed_supports=[[],["a"],["b"],["a","b"]])
    assert report["compression_omitted_eligible_atoms"]==["b"]
    assert report["preferred_supports"]==[["a"],[]]
    assert ["a","b"] in report["heuristic_eligible_supports"]
    assert not report["joint_coverage_calibrated"] and not report["hard_pruning_authorized"]
    assert report["statistically_excluded_supports"]==[]


def test_unknown_atoms_or_unadmitted_selected_support_rejected():
    for selected in ({"support":["unknown"],"one_standard_error_supports":[["unknown"]]},
        {"support":["a"],"one_standard_error_supports":[[]]}):
        with pytest.raises(ValueError):
            summarize_passive_supports(per_scale={"s":{"selected":selected}},
                candidate_names=["a"],allowed_supports=[[],["a"]])


def test_known_coefficients_and_normalized_residuals():
    coefficient=known_coefficients("carbon_evolution",["steam_arrhenius_truth","product_feedback","oxygen_arrhenius_truth"])
    assert np.allclose(coefficient,[-.42,0,-.11])
    matrix=np.eye(3);target=matrix@coefficient
    for factor in (1,9):
        summary=balance_summary(factor*matrix,factor*target,coefficient)
        assert summary["relative_residual_to_target"]==0
        assert summary["target_to_known_source_norm"]==pytest.approx(1)


def test_saved_diagnostics_bound_inputs_and_never_refit():
    for directory in ("nested_passive_observation_balance","nested_passive_preprocessing_stages"):
        payload=json.loads((ROOT/f"artifacts/_audits/{directory}/audit.json").read_text(encoding="utf-8"))
        assert payload["complete"]
        assert payload["new_simulations"]==payload["new_noise_draws"]==payload["new_model_fits"]==0
        for relative,digest in payload["input_hashes"].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    report=json.loads((ROOT/"artifacts/_audits/nested_passive_reporting_contract/audit.json").read_text(encoding="utf-8"))
    assert report["original_selection_unchanged"]
    assert "oxygen_arrhenius_truth" in report["records"]["five_tasks"]["carbon_evolution"]["compression_omitted_eligible_atoms"]
