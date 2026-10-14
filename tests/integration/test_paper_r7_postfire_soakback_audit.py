import json
from pathlib import Path


def test_r7_uses_inherited_natural_cooldowns_and_reports_remaining_aliases():
    root = Path(__file__).resolve().parents[2]
    metrics = json.loads(
        (root / "artifacts/paper_r7_postfire_soakback_audit/metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert all(metrics["physical_checks"].values())
    chemistry_crack = metrics["clean_combined_multiscale_change"][
        "crack_evolution"
    ]["chemistry_stress_truth"]
    assert chemistry_crack["change"] > 0.05

    transport = metrics["clean_combined_multiscale_change"][
        "transport_modulation_proxy"
    ]["combined_opening_truth"]
    assert transport["after_weak_separation"] < 0.01
    assert transport["nearest_competitor_after"] == "stress_only"
