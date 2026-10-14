"""Comparison bookkeeping checks; synthetic metrics, no scientific fits."""
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "reproducibility/compare_h3_precision_estimators.py"
SPEC = importlib.util.spec_from_file_location("precision_comparison", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_audits():
    rows = []
    for role in ("partial", "independent"):
        for seed in range(42301, 42305):
            rows.append({"role": role, "seed": seed, "signature": "test", "input_packet_sha256": f"{role}-{seed}",
                **{m: .1 for m in MODULE.METRICS}, "parameter_gate_pass": True,
                "outer_coefficient_gate_pass": True, "mechanism_action_gate_pass": True,
                "optimizer_success": False, "optimizer_message": "budget", "iterations": 200,
                "function_evaluations": 230, "initial_nll": 1., "final_nll": .9,
                "oracle_nll_post_fit_only": 1.1, "elapsed_seconds": 120.})
    moment_rows = [{**r, "operator": op} for op in ("legacy_simpson", "moment_corrected_degree4") for r in rows]
    return [dict(complete=True, signature="test", records=rr) for rr in (moment_rows, copy.deepcopy(rows), copy.deepcopy(rows))]


def test_same_packets_not_counted_as_new_samples_and_unconverged_retained():
    result = MODULE.compare_saved(*fixture_audits())
    assert result["existing_observation_packets"] == 8
    assert result["conditional_fits_across_three_developments"] == 32
    assert result["likelihood_optimizer_converged"] == 0
    assert len(result["cases"]) == 8 and len(result["summary"]) == 8


@pytest.mark.parametrize("fault", ["incomplete", "missing", "duplicate", "identity", "signature"])
def test_invalid_comparisons_are_rejected(fault):
    audits = fixture_audits()
    last = audits[-1]
    if fault == "incomplete":
        last["complete"] = False
    elif fault == "missing":
        last["records"].pop()
    elif fault == "duplicate":
        last["records"][-1] = copy.deepcopy(last["records"][0])
    elif fault == "identity":
        last["records"][0]["input_packet_sha256"] = "different"
    else:
        last["records"][0]["signature"] = "different"
    with pytest.raises(ValueError):
        MODULE.compare_saved(*audits)


def test_pass_gains_losses_and_error_improvements_are_separate():
    audits = fixture_audits()
    audits[0]["records"][8]["outer_coefficient_gate_pass"] = False
    audits[-1]["records"][1]["parameter_gate_pass"] = False
    audits[-1]["records"][0][MODULE.METRICS[0]] = .05
    result = MODULE.compare_saved(*audits)
    paired = result["paired"][-1]
    assert paired["quantitative_pass_gained"] == 1
    assert paired["quantitative_pass_lost"] == 1
    assert paired[f"improved_{MODULE.METRICS[0]}"] == 1
