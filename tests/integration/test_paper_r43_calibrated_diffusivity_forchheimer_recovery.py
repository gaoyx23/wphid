import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    ROOT
    / "artifacts/paper_r43_calibrated_diffusivity_forchheimer_recovery/metrics.json"
)


def test_r43_reports_conditional_law_topology_and_alpha_recovery():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["candidate_dimensional_alpha_grid"][0] == 0.0
    assert payload["fixed_physical_inputs"][
        "physical_closed_diffusivity_by_species"
    ]
    assert payload["fixed_physical_inputs"][
        "frozen_effective_weak_amplitude_by_species"
    ]
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
        "law_topology_identified_set" in row and "alpha_identified_set" in row
        for block in payload["results"].values()
        for row in block["locked_confirmation"]
    )
    aggregate = payload["aggregate"]
    assert aggregate["in_library_isotropic"]["confirmation_truth_coverage_count"] == 4
    assert aggregate["out_of_library_anisotropic"]["confirmation_unique_truth_count"] == 4
    assert (
        aggregate["out_of_library_Darcy_Forchheimer"][
            "confirmation_truth_coverage_count"
        ]
        == 3
    )
    for row in payload["results"]["out_of_library_Darcy_Forchheimer"][
        "locked_confirmation"
    ]:
        assert all(
            law == "Forchheimer" for law, _ in row["law_topology_identified_set"]
        )
        assert row["nearest_alpha_relative_error_scoring_only"] > 5.0
