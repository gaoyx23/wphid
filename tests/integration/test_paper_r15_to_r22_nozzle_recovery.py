import json
from pathlib import Path


def _load(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def test_sparse_nominal_data_remains_set_valued_but_covers_truth():
    root = Path(__file__).resolve().parents[2]
    r15 = _load(
        root,
        "artifacts/paper_r15_passive_noise_identifiability_boundary/metrics.json",
    )

    nominal = r15["aggregate"]["1.0"]
    for equation in ("carbon_evolution", "crack_evolution"):
        assert nominal[equation]["identified_set_truth_coverage_rate"] == 1.0
        assert nominal[equation]["unique_identification_rate"] == 0.0


def test_fullfield_consensus_recovers_all_five_fixed_library_mechanisms():
    root = Path(__file__).resolve().parents[2]
    r18 = _load(
        root,
        "artifacts/paper_r18_fullfield_transport_capability_control/metrics.json",
    )
    r19 = _load(
        root,
        "artifacts/paper_r19_fullfield_five_mechanism_recovery/metrics.json",
    )
    r20 = _load(
        root,
        "artifacts/paper_r20_cross_scale_consensus_five_mechanisms/metrics.json",
    )

    assert r18["result"]["admissible_family_set"] == ["combined_opening_truth"]
    assert r19["all_five_truth_mechanisms_present_scoring_only"]
    assert not r19["complete_exact_support_recovery_scoring_only"]
    assert r20["complete_exact_support_recovery_scoring_only"]


def test_fullfield_chemistry_and_cracking_parameters_are_quantitatively_recovered():
    root = Path(__file__).resolve().parents[2]
    r21 = _load(
        root,
        "artifacts/paper_r21_fullfield_nonlinear_parameter_recovery/metrics.json",
    )
    r22 = _load(
        root,
        "artifacts/paper_r22_transport_parameter_ratio_recovery/metrics.json",
    )

    assert r21["results"]["chemistry"]["maximum_relative_error_scoring_only"] < 0.04
    assert r21["results"]["cracking"]["maximum_relative_error_scoring_only"] < 0.02
    assert r22["maximum_relative_gain_error_scoring_only"] > 0.40
