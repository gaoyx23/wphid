import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_r73_fresh_development_seed_is_rescued_by_predeclared_global_scale():
    payload = _read(
        "artifacts/paper_r73_multiscale_sparse_transport_development/development.json"
    )
    truth = "combined_opening_truth"
    assert truth not in payload["scales"]["intermediate"]["transport"]["identified_set"]
    assert truth in payload["scales"]["global"]["transport"]["identified_set"]
    assert truth in payload["safe_union_identified_set"]
    assert payload["safe_union_unique_truth_scoring_only"] is False
    assert payload["development_gate"] is True


def test_r74_locked_confirmation_preserves_truth_but_not_uniqueness():
    payload = _read(
        "artifacts/paper_r74_multiscale_sparse_transport_confirmation/metrics.json"
    )
    aggregate = payload["aggregate"]
    assert aggregate["replicate_count"] == 4
    assert aggregate["safe_union_truth_coverage_count"] == 4
    assert aggregate["all_five_truth_coverage_count"] == 4
    assert aggregate["safe_union_unique_truth_count"] == 0
    assert aggregate["mean_safe_union_size"] == 5.25
    assert aggregate["maximum_safe_union_size"] == 6


def test_r75_scale_pooling_is_stopped_when_it_does_not_narrow_the_set():
    payload = _read(
        "artifacts/paper_r75_scale_pooled_transport_development/development.json"
    )
    pooled = payload["scale_pooled"]
    assert pooled["truth_covered_scoring_only"] is True
    assert len(pooled["identified_set"]) == len(payload["safe_union_identified_set"])
    assert payload["development_gate"] is False
