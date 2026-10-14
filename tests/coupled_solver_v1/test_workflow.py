"""New orchestration/statistical bookkeeping checks, no scientific data."""

import json
from pathlib import Path

from wphid.coupled_solver_v1 import Candidate, parameter_box
from reproducibility.coupled_solver_v1.workflow import errors_for_truth, PROTOCOL
from reproducibility.coupled_solver_v1.generator import truth_point, observe
import numpy as np


def test_frozen_plan_has_negative_roles_complete_counts_and_no_native_claim():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert {"null","oxygen_only","steam_only","both_free","both_independent","near_equivalent"}.issubset(p["data"]["source_roles"])
    assert p["data"]["observation_block_count"]==len(p["data"]["source_roles"])*p["data"]["noise_blocks_per_role"]
    assert p["data"]["physical_trajectory_count"]==len(p["data"]["source_roles"])*p["data"]["mission_count"]
    assert p["external_adapter"]["no_postfit_profile"]


def test_exact_artificial_parameters_are_same_fit_precision_but_wrong_structure_fails():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"));point,a=truth_point("both_shared")
    c=Candidate("both_shared");params={k:point[k] for k in parameter_box(c)}
    fit={"fitted":True,"parameters":params,"coefficients":a}
    truth={"role":"both_shared","source":"both_shared","parameters":point,"coefficients":a}
    result=errors_for_truth(c,fit,truth,p["precision"])
    assert result["reaction_score"]==0 and result["joint_structure_and_precision"]
    wrong=errors_for_truth(Candidate("both_swapped_product"),fit,truth,p["precision"])
    assert wrong["precision_pass"] and not wrong["joint_structure_and_precision"]


def test_failed_estimate_never_becomes_success_or_missing_denominator():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result=errors_for_truth(Candidate("both_shared"),{"fitted":False},{},p["precision"])
    assert not result["precision_pass"] and not result["assessable"]


def test_fixed_initial_and_positive_zero_sensor_law_are_explicit():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"));shape=(3,2,2)
    fields={"T":np.full(shape,330.),"carbon":np.full(shape,.95),"pO2":np.full(shape,100.),"pH2O":np.full(shape,1000.),
        "pProducts":np.zeros(shape),"crack":np.full(shape,.02),"stress":np.full(shape,1e7)}
    observed,loads,sigma,load_sigma,receipt=observe(fields,{"surface_heat_flux":np.full((3,2),100.)},np.random.default_rng(991),p["observation_contract"])
    for key in fields:
        np.testing.assert_array_equal(observed[key][0],fields[key][0]);assert np.all(sigma[key][0]==0)
    assert np.all(observed["pProducts"]==0) and np.all(sigma["pProducts"]==0)
    assert receipt["clipping_or_reconstruction"] is False
    assert np.any(load_sigma["surface_heat_flux"][0]>0)


def test_inactive_nuisance_coordinates_are_retained_in_full_parameter_box():
    box=parameter_box(Candidate("both_shared"))
    assert "thermal.modulation.stiffness_floor" in box
    assert "thermal.modulation.conductivity_floor" in box
    assert "gas.modulation.total_cap" in box
