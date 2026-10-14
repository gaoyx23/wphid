"""All new directions stopped before refitting; provenance checks only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "artifacts/_audits/h3_nonlinear_amplitude_profile"
ENRICHMENT = ROOT / "artifacts/_audits/h3_test_enrichment"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_mean_bias_is_one_term_and_not_mislabeled_a_full_estimator():
    audit = read(PROFILE / "mean-bias-diagnostic.json")
    assert audit["complete"] and audit["optimizations"] == 0 and not audit["truth_used"]
    assert all(r["do_not_refit_this_direction"] for r in audit["records"])
    assert "Does not exclude all EIV bias" in audit["claim_boundary"]


def test_more_tests_do_not_count_as_more_observations_or_accuracy_results():
    audit = read(ENRICHMENT / "audit.json")
    assert audit["complete"] and audit["optimizations"] == audit["new_observations"] == 0
    assert all(not r["worth_minimal_fit_check"] for r in audit["records"])
    for row in audit["records"]:
        assert [r["weak_tests_per_condition"] for r in row["variants"]] == [27, 108]
        assert max(row["variants"][1]["clean_balance_relative_errors"]) < 1e-4


def test_resolution_witness_not_declared_confidence_or_truth_selection():
    audit = read(PROFILE / "resolution-witness.json")
    assert audit["complete"] and audit["optimizations"] == 0 and not audit["truth_used"]
    assert not audit["statistical_budget_calibrated"]
    assert all(r["witness"] is not None for r in audit["records"])


def test_new_sources_and_existing_profiles_stay_frozen():
    for path in (PROFILE / "mean-bias-diagnostic.json", PROFILE / "resolution-witness.json", ENRICHMENT / "protocol.json"):
        for source, expected in read(path)["source_sha256"].items():
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected
