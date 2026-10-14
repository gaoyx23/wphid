import ast
import json
from pathlib import Path
from experiments.organize_research_evidence import ROOT, group, resolve


def test_paths_resolve_statically_without_running_code():
    expression=ast.parse('ROOT / "artifacts" / "example"',mode="eval").body
    assert resolve(expression,{"ROOT":ROOT})==ROOT/"artifacts/example"
    assert resolve(ast.parse('simulate()',mode="eval").body,{}) is None


def test_physical_and_historical_baseline_navigation():
    assert group("paper_r110_external_structural_audit").startswith("01")
    assert group("paper_b2_native_cartesian_baselines").startswith("06")
    assert group("paper_r20_cross_scale_consensus_five_mechanisms").startswith("02")


def test_inventory_covers_sources_and_does_not_call_counts_experiments():
    saved=json.loads((ROOT/"docs/research/experiment-inventory.json").read_text(encoding="utf-8"))
    expected={str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT/"experiments").rglob("*.py")}
    assert {r["script"] for r in saved["sources"]}==expected
    assert saved["summary"]["counts_are_inventory_not_scientific_experiment_counts"]
    pending=[r for r in saved["sources"] if "paper_r170_" in r["script"] or "paper_r171_" in r["script"]]
    assert len(pending)==2 and all("尚未运行" in r["state"] for r in pending)


def test_saved_audit_and_cache_preflight_are_not_new_confirmation():
    base=ROOT/"artifacts/_audits/research_mainline_review"
    report=json.loads((base/"audit.json").read_text(encoding="utf-8"))
    preflight=json.loads((base/"next-panel-preflight.json").read_text(encoding="utf-8"))
    assert report["new_simulations"]==report["new_noise_draws"]==report["new_model_fits"]==0
    assert report["r106_r107_not_controlled_data_addition"]["decision_rule_changed"]
    assert preflight["ready"] and preflight["mission_count"]==13
    assert not preflight["recovery_panel_started"] and not preflight["scientific_protocol_frozen"]
