import json

from reproducibility.audit_joint_recovery_bottlenecks import SOURCE, summarize


def test_saved_confirmation_separates_estimation_and_reporting():
    report = summarize(json.loads(SOURCE.read_text(encoding="utf-8"))["records"])
    groups = {r["role"]: r for r in report["groups"] if r["split"] == "confirmation"}
    assert [groups[k]["pooled_quantitative_pass"] for k in ("shared", "partial", "independent")] == [10, 8, 4]
    assert [groups[k]["full_output"] for k in ("shared", "partial", "independent")] == [10, 1, 0]
    assert sum(r["quantitative_pass_withheld"] for r in groups.values()) == 11
    assert sum(r["estimation_failure"] for r in groups.values()) == 8
    assert all(r["structural_endpoint_correct"] == 10 for r in groups.values())
    assert all(r["inaccurate_full_output"] == 0 for r in groups.values())


def test_development_is_not_pooled_with_confirmation():
    report = summarize(json.loads(SOURCE.read_text(encoding="utf-8"))["records"])
    groups = [r for r in report["groups"] if r["split"] == "development"]
    assert [r["count"] for r in groups] == [4, 4, 4]
    assert [r["pooled_quantitative_pass"] for r in groups] == [4, 3, 3]
    assert "not fresh validation" in report["interpretation"]["confirmation_reuse"]
