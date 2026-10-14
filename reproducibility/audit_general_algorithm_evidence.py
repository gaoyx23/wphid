"""Read saved algorithm records only: no experiment imports, fits or simulation."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/research/algorithm-evidence"
SOURCES = {
    "h1": "artifacts/paper_h1_heterogeneous_three_class_network/metrics.json",
    "h2_old": "artifacts/paper_h2_optional_sharing_controls/metrics.json",
    "h2": "artifacts/paper_h2_partition_stability/metrics.json",
    "h3": "artifacts/paper_h3_joint_family_topology_partition/metrics.json",
    "h3_tier": "artifacts/paper_h3_v6_identifiability_tier/metrics.json",
    "joint": "artifacts/s3b_e4_joint_family_topology/metrics.json",
    "independent": "artifacts/paper_b1_independent_weak_baseline/metrics.json",
    "pointwise": "artifacts/paper_b1_standard_sindy/metrics.json",
    "cartesian": "artifacts/paper_b2_native_cartesian_baselines/metrics.json",
    "implicit": "artifacts/paper_b3_official_sindypi/metrics.json",
    "ablations": "artifacts/paper_a2_a3_ablation/metrics.json",
    "runtime": "artifacts/paper_b4_runtime_closed_loop/metrics.json",
}
SOURCE_CODE = (
    "experiments/paper_h1_heterogeneous_three_class_network.py",
    "experiments/paper_h2_partition_stability.py",
    "experiments/paper_h3_joint_family_topology_partition.py",
    "experiments/s3b_e1_closed_coupled_network.py",
    "experiments/paper_b1_independent_weak_baseline.py",
    "experiments/paper_b1_standard_sindy.py",
    "experiments/paper_b2_native_cartesian_baselines.py",
    "experiments/paper_a2_a3_ablation.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = {name: json.loads((ROOT / path).read_text(encoding="utf-8"))
            for name, path in SOURCES.items()}
    hashes = {path: digest(ROOT / path) for path in (*SOURCES.values(), *SOURCE_CODE)}
    raw = []

    def record(source, pointer, value):
        raw.append(dict(source=SOURCES[source], source_sha256=hashes[SOURCES[source]],
                        json_pointer=pointer, value=value))

    for name, saved in data.items():
        for key in ("protocol", "aggregate", "confirmation_summary", "confirmation_by_role",
                    "positive_exact_rate", "negative_exact_rate", "timing_summary", "forecast_summary",
                    "a2_no_condition_stability", "a3_no_output_tiering"):
            if key in saved:
                record(name, "/" + key, saved[key])
    for index, method in enumerate(data["cartesian"]["methods"]):
        for key in ("method", "positive_exact_rate", "negative_exact_rate"):
            record("cartesian", f"/methods/{index}/{key}", method[key])

    h1_rows = [row for row in data["h1"]["records"] if row["split"] == "confirmation"]
    h1_counts = []
    for role in sorted({row["role"] for row in h1_rows}):
        rows = [row for row in h1_rows if row["role"] == role]
        h1_counts.append(dict(role=role, count=len(rows),
            compatible_family_and_topology=sum(row["overall_exact"] for row in rows),
            exact_family=sum(row["complete_family_exact"] for row in rows),
            exact_structural_output_with_margin=sum(row["complete_equation_success"] for row in rows)))
    for index, row in enumerate(data["h1"]["records"]):
        if row["split"] == "confirmation":
            for key in ("split", "seed", "role", "overall_exact", "complete_family_exact", "complete_equation_success"):
                record("h1", f"/records/{index}/{key}", row[key])

    paired = []
    truth_source = "experiments/s3b_e1_closed_coupled_network.py"
    tree = ast.parse((ROOT / truth_source).read_text(encoding="utf-8"))
    truth = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PROTOCOL"
            for target in node.targets))
    independent_by_seed = {row["seed"]: row for row in data["independent"]["confirmation_positive"]}
    for index, row in enumerate(data["joint"]["confirmation_positive"]):
        best = row["refined_ranking"][0]
        record("joint", f"/confirmation_positive/{index}/refined_ranking/0", best)
        parameter_error = max(abs(estimate - actual) / abs(actual)
            for estimate, actual in zip(best["parameters"], truth["truth_parameters"]))
        coefficient_error = max(abs(estimate - actual) / abs(actual)
            for field, estimates in zip(("temperature", "oxygen", "damage"), best["coefficients"])
            for estimate, actual in zip(estimates, truth["truth_coefficients"][field]))
        other = independent_by_seed[row["seed"]]
        paired.append(dict(seed=row["seed"], joint_parameter_max=parameter_error,
            joint_coefficient_max=coefficient_error,
            independent_parameter_max=other["maximum_parameter_relative_error"],
            independent_coefficient_max=other["maximum_coefficient_relative_error"]))

    seeds = {}
    for name in ("h1", "h2", "h3", "h3_tier"):
        protocol = data[name]["protocol"]
        development = set(protocol["development_seeds"])
        confirmation = set(protocol["confirmation_seeds"])
        seeds[name] = dict(development_seed_count=len(development), confirmation_seed_count=len(confirmation),
            seed_overlap=sorted(development & confirmation),
            confirmation_locked_explicit=protocol.get("confirmation_locked"),
            disjoint_seeds_are_not_proof_of_no_adaptive_reuse=True)
    seeds["h2"]["confirmation_overlap_with_old_v2"] = sorted(
        set(data["h2"]["protocol"]["confirmation_seeds"]) &
        set(data["h2_old"]["protocol"]["confirmation_seeds"]))
    for name in ("independent", "pointwise", "joint", "cartesian", "implicit"):
        protocol = data[name]["protocol"]
        seeds[name] = {key: protocol[key] for key in (
            "development_positive_seeds", "development_negative_seeds",
            "confirmation_positive_seeds", "confirmation_negative_seeds")}
    assert len(h1_rows) == 50
    assert sum(row["overall_exact"] for row in h1_rows) == 49
    assert sum(row["complete_equation_success"] for row in h1_rows) == 45
    assert all(not seeds[name]["seed_overlap"] for name in ("h1", "h2", "h3", "h3_tier"))
    assert seeds["h2"]["confirmation_overlap_with_old_v2"] == []
    assert len(paired) == 5
    report = dict(role="read_only_saved_algorithm_evidence_audit", new_experiments=0,
        new_model_fits=0, new_weak_projections=0, source_hashes=hashes,
        raw_source_fields=raw, h1_derived_counts=h1_counts, paired_shared_parameter_errors=paired,
        paired_error_truth_reference=dict(source=truth_source, source_sha256=hashes[truth_source],
            static_ast_literal_assignment="PROTOCOL", truth_parameters=truth["truth_parameters"],
            truth_coefficients=truth["truth_coefficients"]),
        seed_checks=seeds,
        interpretations=dict(h1_complete_equation_success_is_not_parameter_accuracy=True,
            h1_protocol_explicitly_not_locked=True,
            h3_is_one_reaction_family_partition_search_not_three_class_coexistence=True,
            native_cartesian_baselines_not_same_dataset_as_axisymmetric_joint=True,
            ablations_are_saved_record_reinterpretation_not_retrained_models=True))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "audit.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    manifest = dict(generator="reproducibility/audit_general_algorithm_evidence.py",
        generator_sha256=digest(Path(__file__)), source_hashes=hashes,
        output="docs/research/algorithm-evidence/audit.json", output_sha256=digest(path),
        original_results_modified=False, new_scientific_experiments=0)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(h1_counts=h1_counts, paired_errors=paired, h1_locked=seeds["h1"]["confirmation_locked_explicit"],
        h2_old_confirmation_overlap=seeds["h2"]["confirmation_overlap_with_old_v2"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
