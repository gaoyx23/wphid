import json
from reproducibility.audit_general_algorithm_evidence import ROOT, OUT, digest
from reproducibility.assemble_closeout_evidence import pointer


def report():
    return json.loads((OUT / "audit.json").read_text(encoding="utf-8"))


def test_raw_algorithm_values_and_source_hashes():
    saved = report()
    assert saved["new_experiments"] == saved["new_model_fits"] == saved["new_weak_projections"] == 0
    sources = {}
    for row in saved["raw_source_fields"]:
        path = ROOT / row["source"]
        assert digest(path) == row["source_sha256"]
        if row["source"] not in sources:
            sources[row["source"]] = json.loads(path.read_text(encoding="utf-8"))
        assert pointer(sources[row["source"]], row["json_pointer"]) == row["value"]


def test_h1_counts_not_mislabeled_as_joint_parameter_recovery():
    saved = report()
    counts = saved["h1_derived_counts"]
    assert sum(row["count"] for row in counts) == 50
    assert sum(row["compatible_family_and_topology"] for row in counts) == 49
    assert sum(row["exact_structural_output_with_margin"] for row in counts) == 45
    assert saved["seed_checks"]["h1"]["confirmation_locked_explicit"] is False
    assert saved["interpretations"]["h1_complete_equation_success_is_not_parameter_accuracy"]


def test_paired_parameter_improvement_not_all_coefficient_cases_win():
    paired = report()["paired_shared_parameter_errors"]
    assert len(paired) == 5
    assert all(row["joint_parameter_max"] < row["independent_parameter_max"] for row in paired)
    assert sum(row["joint_coefficient_max"] < row["independent_coefficient_max"] for row in paired) == 4
    assert round(100 * max(row["joint_parameter_max"] for row in paired), 2) == 4.79
    assert round(100 * max(row["joint_coefficient_max"] for row in paired), 2) == 17.26


def test_seed_isolation_and_baseline_task_separation():
    saved = report()
    for name in ("h1", "h2", "h3", "h3_tier"):
        assert saved["seed_checks"][name]["seed_overlap"] == []
    assert saved["seed_checks"]["h2"]["confirmation_overlap_with_old_v2"] == []
    assert saved["seed_checks"]["joint"]["confirmation_positive_seeds"] != saved["seed_checks"]["cartesian"]["confirmation_positive_seeds"]
    assert saved["interpretations"]["native_cartesian_baselines_not_same_dataset_as_axisymmetric_joint"]


def test_manifest_binds_generator_sources_and_export():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["new_scientific_experiments"] == 0
    assert digest(ROOT / manifest["generator"]) == manifest["generator_sha256"]
    assert digest(ROOT / manifest["output"]) == manifest["output_sha256"]
    for path, expected in manifest["source_hashes"].items():
        assert digest(ROOT / path) == expected
