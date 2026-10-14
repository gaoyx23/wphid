import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "paper_r109_external_fea_contract_preflight"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r109_is_explicitly_an_interface_preflight_not_external_confirmation():
    metrics = json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))
    contract = json.loads((ARTIFACT / "contract.json").read_text(encoding="utf-8"))

    assert metrics["preflight_complete"] is True
    assert metrics["external_solver_executed"] is False
    assert metrics["external_solver_result_available"] is False
    assert metrics["external_validation_claim_authorized"] is False
    assert contract["scientific_status"] == "preflight_only_external_execution_pending"
    assert contract["scientific_comparison_authorized"] is False
    assert contract["source"]["trajectory_regenerated"] is False


def test_r109_nozzle_and_analytic_qualification_decks_pass_preflight():
    contract = json.loads((ARTIFACT / "contract.json").read_text(encoding="utf-8"))

    assert contract["nozzle_deck_audit"]["passed"] is True
    assert contract["nozzle_deck_audit"]["all_pressure_faces_are_p4"] is True
    assert contract["nozzle_deck_audit"]["all_inner_faces_loaded_once"] is True
    assert contract["qualification"]["lame_pressure"]["deck_audit"]["passed"] is True
    assert (
        contract["qualification"]["uniform_free_heating"]["deck_audit"]["passed"]
        is True
    )
    assert (
        contract["qualification"]["uniform_axial_restraint_heating"]["deck_audit"]["passed"]
        is True
    )
    assert (
        contract["qualification"]["lame_mesh_refinement"]["coarse_deck_audit"]["passed"]
        is True
    )
    assert contract["mesh"]["node_count"] == 441
    assert contract["mesh"]["element_count"] == 384
    assert contract["mesh"]["source_and_external_resolution_are_different"] is True


def test_r109_exported_files_are_hash_locked_and_label_free():
    contract = json.loads((ARTIFACT / "contract.json").read_text(encoding="utf-8"))
    for name, record in contract["generated_files"].items():
        path = ARTIFACT / name
        assert path.exists()
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]

    forbidden = ("truth", "mechanism", "support", "coefficient", "candidate_family")
    for name in (
        "mesh_nodes.csv",
        "mesh_elements.csv",
        "nodal_temperature.csv",
        "inner_pressure.csv",
    ):
        with (ARTIFACT / name).open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        lowered = " ".join(header).lower()
        assert not any(token in lowered for token in forbidden)


def test_r109_source_hash_points_to_the_frozen_r100_snapshot():
    contract = json.loads((ARTIFACT / "contract.json").read_text(encoding="utf-8"))
    source = ROOT / contract["source"]["relative_path"]

    assert source.exists()
    assert _sha256(source) == contract["source"]["sha256"]
    assert contract["source"]["snapshot_time_s"] == 888.0
