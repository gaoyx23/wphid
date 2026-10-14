import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/paper_r42_species_profiled_forchheimer_recovery/metrics.json"


def test_r42_reports_species_laws_topology_and_dimensional_alpha_separately():
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["candidate_profile_numbers_unchanged_from_R41"] == [
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
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
        set(row["species_profile_results"]) == {"pO2", "pH2O", "pProducts"}
        and "topology_identified_set" in row
        and "profiled_dimensional_alpha_by_species" in row
        for block in payload["results"].values()
        for row in block["locked_confirmation"]
    )
    aggregate = payload["aggregate"]
    assert aggregate["in_library_isotropic"]["confirmation_topology_coverage_count"] == 4
    assert aggregate["out_of_library_anisotropic"]["confirmation_topology_coverage_count"] == 4
    for species in ("pO2", "pH2O"):
        assert (
            aggregate["out_of_library_Darcy_Forchheimer"][
                "confirmation_species_law_coverage_count"
            ][species]
            == 4
        )
    assert (
        aggregate["out_of_library_Darcy_Forchheimer"][
            "confirmation_species_law_coverage_count"
        ]["pProducts"]
        == 0
    )
    assert (
        aggregate["out_of_library_Darcy_Forchheimer"][
            "confirmation_all_species_law_coverage_count"
        ]
        == 0
    )
