import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    ROOT / "artifacts/paper_r41_profiled_forchheimer_family_recovery/metrics.json"
)


def test_r41_reports_law_topology_and_profile_separately():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["candidate_profile_numbers"][0] == 0.0
    assert set(payload["results"]) == {
        "in_library_isotropic",
        "out_of_library_anisotropic",
        "out_of_library_Darcy_Forchheimer",
    }
    assert all(
        len(block["locked_confirmation"]) == 4
        for block in payload["results"].values()
    )
    assert all(
        "law_topology_identified_set" in row
        and "profile_number_identified_set" in row
        for block in payload["results"].values()
        for row in block["locked_confirmation"]
    )
    aggregate = payload["aggregate"]
    assert aggregate["in_library_isotropic"]["confirmation_unique_truth_count"] == 4
    assert (
        aggregate["out_of_library_anisotropic"][
            "confirmation_unique_truth_count"
        ]
        == 4
    )
    assert (
        aggregate["out_of_library_Darcy_Forchheimer"][
            "confirmation_truth_coverage_count"
        ]
        == 0
    )
