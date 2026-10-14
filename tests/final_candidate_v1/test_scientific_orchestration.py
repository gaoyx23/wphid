"""Literal saved-record/job checks; no observation generation or fitting."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproducibility/final_candidate_v1"))
from run_confirmation import PROTOCOL, jobs_for
from assess_confirmation import assess_fit
from experiments import paper_h3_joint_family_topology_partition as reference


def literal_record(role):
    case = reference.TRUTH_CASES[role]
    groups = case["partition"]
    parameters = [case["parameters"][groups.index(group)] for group in sorted(set(groups))]
    coefficients = [list(reference.h1.CLOSED["truth_coefficients"][field][:-1])+[amplitude]
                    for field,amplitude in zip(reference.FIELDS,reference.TRUTH_AMPLITUDES)]
    return {"status":"success","fit":{"hypothesis":[{"family":reference.TRUTH_FAMILY,"groups":list(groups),
            "internal_parameters_by_group":parameters}],"physical_coefficients":coefficients,
            "optimizer_status":"budget_exhausted","cpu_seconds":1.},
            "frozen_heldout":{"equation_weak_target_relative_residual":[.1,.2,.3]}}


def test_jobs_keep_all_blocks_and_complete_open_candidate_space():
    inputs = {"records":[{"stage":stage,"seed":seed,"role":role,"file":"literal.npz","sha256":"literal"}
                         for stage in ("Q1","Q3") for seed in PROTOCOL[stage]["block_seeds"]
                         for role in PROTOCOL[stage]["roles"]]}
    counts = []
    for stage in ("Q1","Q2","Q3"):
        jobs = jobs_for(stage,inputs)
        counts.append(len(jobs))
        assert len({j["id"] for j in jobs}) == len(jobs)
    assert counts == [108,36,645]


def test_sharing_controls_never_refit_the_eighteen_correct_q1_records():
    inputs = {"records":[{"stage":"Q1","seed":seed,"role":role}
                         for seed in PROTOCOL["Q1"]["block_seeds"] for role in PROTOCOL["Q1"]["roles"]]}
    for job in jobs_for("Q2",inputs):
        assert job["partition_name"] != job["input"]["role"]
        assert job["groups"] != PROTOCOL["Q1"]["known_partition_by_role"][job["input"]["role"]]
        assert job["input"]["seed"] in PROTOCOL["Q2"]["reuse_Q1_block_seeds"]


def test_partial_group_errors_are_expanded_to_the_correct_destinations():
    result = assess_fit(literal_record("partial"),"partial")
    assert result["precision_pass"] and result["exact_family_topology_partition"]
    assert result["case_score"] == 0
    np.testing.assert_array_equal(result["parameter_errors_by_equation"],np.zeros((3,3)))
    assert result["budget_exhausted"]  # Truncation is recorded, not sample removal.


def test_critical_gate_is_simultaneous_and_excludes_base_coefficient_errors():
    record = literal_record("shared")
    record["fit"]["physical_coefficients"][0][0] *= 100
    assert assess_fit(record,"shared")["precision_pass"]
    record["fit"]["hypothesis"][0]["internal_parameters_by_group"] = [[9000.,.65,1.25]]
    result = assess_fit(record,"shared")
    assert not result["precision_pass"] and result["case_score"] > 2


def test_zero_amplitude_or_failed_fit_stays_invalid_not_a_success():
    record = deepcopy(literal_record("independent"))
    record["fit"]["physical_coefficients"][1][-1] = 0
    assert not assess_fit(record,"independent")["assessable"]
    failed = assess_fit({"status":"failed"},"shared")
    assert not failed["precision_pass"] and failed["case_score"] is None


def test_wrong_family_cannot_borrow_true_family_parameter_accuracy():
    record = literal_record("shared")
    record["fit"]["hypothesis"][0]["family"] = "arrhenius_oxygen"
    result = assess_fit(record,"shared")
    assert not result["assessable"] and not result["precision_pass"]
