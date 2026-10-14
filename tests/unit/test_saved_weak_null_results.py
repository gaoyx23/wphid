"""Check saved null-model accounting without reevaluating weak balances."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT/"artifacts/_audits/saved_weak_null_recovery/audit.json"


@pytest.fixture
def data():
    if not RESULT.exists(): pytest.skip("Local saved experiment unavailable")
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_candidate_null_calibration_never_uses_evaluation_indices(data):
    assert data["complete"]
    for models in data["models"].values():
        for name, model in models.items():
            assert model["null_simulation_source"] == name
            assert model["calibration_draws"] == [64,95]
    assert not data["protocol"]["unknown_observation_clean_baseline_subtracted"]


def test_summary_counts_match_retained_sets(data):
    for sources in data["results"].values():
        for source, result in sources.items():
            sets = [v["candidate_set"] for v in result["observations"]]
            summary = result["summary"]
            assert len(sets) == summary["evaluation_count"] == 32
            assert summary["truth_retained"] == sum(source in c for c in sets)
            assert summary["unique_correct"] == sum(c == [source] for c in sets)
            assert summary["wrong_unique"] == sum(len(c)==1 and source not in c for c in sets)
            assert summary["empty"] == sum(not c for c in sets)


def test_true_null_retention_matches_noise_range_check(data):
    multi = json.loads((ROOT/"artifacts/_audits/multisource_finite_noise/audit.json").read_text())
    for method, sources in data["results"].items():
        for source, result in sources.items():
            if source == "Darcy__isotropic": continue
            diagnostic = multi["summaries"][source][source]["methods"][method]
            assert result["summary"]["truth_retained"] == diagnostic["evaluation_inside_count"]
