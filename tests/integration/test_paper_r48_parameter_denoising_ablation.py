import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r48_parameter_denoising_ablation/metrics.json"


def test_r48_documents_the_single_view_topology_parameter_tradeoff():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    development = payload["development_ablation"]
    assert development["raw_clipped"]["alpha_output"] == 0.5
    assert development["temporal_savgol"]["alpha_output"] == 0.5
    assert development["raw_clipped"][
        "unique_truth_law_topology_scoring_only"
    ] is False
    assert payload["selected_method_frozen_before_confirmation"] == (
        "space_time_savgol"
    )
    aggregate = payload["aggregate"]
    assert aggregate["confirmation_truth_coverage_count"] == 4
    assert aggregate["confirmation_internal_alpha_minimum_count"] == 0
    assert aggregate["confirmation_consistent_numeric_alpha"] is False
