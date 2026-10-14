import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r68_long_bell_retains_local_mechanisms_but_transport_is_three_of_four():
    path = ROOT / "artifacts/paper_r68_two_sided_bell_geometry_confirmation/metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    qualification = payload["qualification"]
    assert qualification["length_scale"] == 1.05
    assert qualification["monotone_bell"] is True
    assert qualification["all_conditions_finite"] is True
    aggregate = payload["confirmation_aggregate"]
    assert aggregate["replicate_count"] == 4
    assert aggregate["carbon_truth_coverage_rate"] == 1.0
    assert aggregate["crack_truth_coverage_rate"] == 1.0
    assert aggregate["transport_truth_coverage_rate"] == 0.75
    assert aggregate["all_five_truth_coverage_rate"] == 0.75
    assert aggregate["complete_exact_five_mechanism_rate"] == 0.0
