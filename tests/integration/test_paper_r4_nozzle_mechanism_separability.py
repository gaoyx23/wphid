import json
from pathlib import Path


def test_r4_tracks_r3_and_exposes_remaining_natural_data_aliases():
    root = Path(__file__).resolve().parents[2]
    r3 = json.loads(
        (root / "artifacts/paper_r3_passive_nozzle_dataset/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    r4 = json.loads(
        (root / "artifacts/paper_r4_nozzle_mechanism_separability/metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert r4["parent_r3_protocol_sha256"] == r3["protocol_sha256"]
    limited = next(
        row for row in r4["records"]
        if row["stage"] == "limited_single_mission" and row["noise_level"] == "clean"
    )
    sufficient = next(
        row for row in r4["records"]
        if row["stage"] == "sufficient_four_missions" and row["noise_level"] == "clean"
    )
    assert sum(value == "full" for value in limited["mechanism_observability"].values()) == 0
    assert sum(value == "full" for value in sufficient["mechanism_observability"].values()) == 5
    chemistry = sufficient["weak_candidate_groups"]["chemical_reaction"]
    assert chemistry["truth_candidate_separation"]["oxygen_arrhenius_truth"][
        "nearest_competitor"
    ] == "generic_thermal_carbon"
    assert chemistry["truth_candidate_separation"]["oxygen_arrhenius_truth"][
        "weak_separation"
    ] < 0.01
