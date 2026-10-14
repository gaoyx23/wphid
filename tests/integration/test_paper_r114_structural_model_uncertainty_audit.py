import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "paper_r114_structural_model_uncertainty_audit"
METRICS = OUTPUT / "metrics.json"
FIGURE = OUTPUT / "structural_model_uncertainty.png"


def _load():
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_r114_reuses_nonredundant_existing_models_only():
    metrics = _load()
    assert metrics["new_fea_runs"] == 0
    assert metrics["truth_or_trajectory_regenerated"] is False
    assert metrics["primary_model_names"] == [
        "free_thermal_expansion",
        "axial_roller_attachment",
        "compliant_attachment",
        "sliding_flange",
    ]
    assert set(metrics["limiting_control_names"]) == {
        "fully_clamped_inlet", "fully_bonded_flange"
    }
    assert len(metrics["excluded_from_voting"]) == 4


def test_r114_exposes_crack_location_model_form_uncertainty():
    metrics = _load()
    ensemble = metrics["primary_ensemble"]
    assert ensemble["model_count"] == 4
    assert ensemble["element_count"] == 384
    assert ensemble["any_model_fraction"] > ensemble["all_models_fraction"]
    assert metrics["interpretation"]["threshold_activation_is_model_form_sensitive"] is True
    assert metrics["interpretation"]["primary_consensus_location_supported"] is True
    assert metrics["interpretation"]["unconditional_crack_location_claim_supported"] is False
    assert metrics["interpretation"]["truth_promotion_allowed"] is False


def test_r114_separates_structurally_robust_and_conditional_mechanisms():
    scope = _load()["mechanism_claim_scope"]
    assert "arrhenius_carbon_oxidation_topology" in scope["structurally_robust"]
    assert "stress_assisted_cracking_activation_and_location" in scope["structural_model_conditional"]
    assert "stress_dependent_crack_opening_and_transport_feedback" in scope["structural_model_conditional"]


def test_r114_source_hashes_and_figure_are_present():
    metrics = _load()
    assert FIGURE.stat().st_size > 50_000
    for record in metrics["source_files"].values():
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
