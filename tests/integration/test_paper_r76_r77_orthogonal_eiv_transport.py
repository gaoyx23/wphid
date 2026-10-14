import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r76_eiv_development_gate_passes_with_all_candidates_ranked():
    payload = _read(
        "artifacts/paper_r76_orthogonal_eiv_transport_development/development.json"
    )
    baseline = payload["results"]["nonnegative_least_squares"]
    eiv = payload["results"]["nonnegative_orthogonal_eiv"]
    assert baseline["identified_set_size"] == 6
    assert eiv["identified_set"] == ["combined_opening_truth"]
    assert eiv["ranked_family_count"] == 6
    assert payload["development_gate"] is True


def test_r77_apparent_unique_result_is_invalidated_by_candidate_fit_failures():
    payload = _read(
        "artifacts/paper_r77_orthogonal_eiv_transport_confirmation/metrics.json"
    )
    aggregate = payload["aggregate"]
    assert aggregate["orthogonal_eiv_unique_truth_count"] == 4
    assert aggregate["all_candidates_numerically_ranked_count"] == 1
    assert payload["confirmation_valid"] is False


def test_r78_candidate_complete_guard_preserves_truth_and_prevents_false_unique():
    payload = _read(
        "artifacts/paper_r78_candidate_complete_eiv_guard_confirmation/metrics.json"
    )
    aggregate = payload["aggregate"]
    assert aggregate["replicate_count"] == 4
    assert aggregate["candidate_complete_eiv_count"] == 2
    assert aggregate["fallback_count"] == 2
    assert aggregate["guarded_truth_coverage_count"] == 4
    assert aggregate["guarded_unique_truth_count"] == 2
    assert aggregate["all_five_truth_coverage_count"] == 4
    assert aggregate["mean_same_packet_baseline_set_size"] == 3.25
    assert aggregate["mean_guarded_set_size"] == 2.5
    for row in payload["records"]:
        if row["decision"] == "candidate_complete_eiv":
            assert row["eiv_ranked_family_count"] == 6
        else:
            assert row["guarded_set"] == row["baseline_set"]
