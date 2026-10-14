"""Read saved H3 provenance only; never import experiment or wphid modules.

Print JSON to stdout. This tool neither rebuilds arrays nor writes results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_key(record):
    return (record["split"], record["role"], record["seed"])


def inspect_workers(result_path, directories):
    aggregate = read_json(result_path)
    aggregate_records = {record_key(r): r for r in aggregate["records"]}
    found = {}
    schemas = Counter()
    mismatches, hash_mismatches, duplicates = [], [], []
    source_hashes = {str(result_path.relative_to(ROOT)).replace("\\", "/"): sha256(result_path)}
    folded = Counter()
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected saved worker directory is missing: {directory}")
        for path in sorted(directory.glob("*.json")):
            payload = read_json(path)
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            source_hashes[relative] = sha256(path)
            schemas[tuple(sorted(payload))] += 1
            record = payload["record"]
            key = record_key(record)
            if key in found:
                duplicates.append(list(key))
            found[key] = relative
            if aggregate_records.get(key) != record:
                mismatches.append(relative)
            if payload.get("protocol_sha256") != aggregate["protocol_sha256"]:
                hash_mismatches.append(relative)
            choices = record.get("partition_evidence", {}).get("fold_choices", [])
            if choices:
                folded[(record["split"], len(choices), len(set(choices)))] += 1
    missing = [list(key) for key in aggregate_records if key not in found]
    extra = [list(key) for key in found if key not in aggregate_records]
    return {
        "aggregate": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "aggregate_count": len(aggregate["records"]),
        "worker_count": len(source_hashes) - 1,
        "worker_schemas": [{"keys": list(keys), "count": n} for keys, n in schemas.items()],
        "record_mismatches": mismatches,
        "protocol_hash_mismatches": hash_mismatches,
        "duplicates": duplicates,
        "missing_records": missing,
        "extra_records": extra,
        "fold_choice_counts": [
            {"split": split, "fold_count": n, "distinct_choices": k, "records": count}
            for (split, n, k), count in sorted(folded.items())
        ],
        "aggregate_decision": aggregate["aggregate"]["decision"],
        "source_sha256": source_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path,
                        help="Optional read-only root for exactly bound files and H3 README")
    args = parser.parse_args()
    binding = read_json(ROOT / "docs/research/algorithm-evidence/final-config-static-binding.json")
    checks = {
        path: sha256(ROOT / path) == expected
        for path, expected in binding["source_sha256"].items()
    }
    manifest_checks = {}
    for relative, sections in (
        ("reproducibility/workspace-baseline-manifest.json", ("included",)),
        ("reproducibility/canonical_manifest.json", ("entry_points", "supporting_code", "artifacts")),
    ):
        document = read_json(ROOT / relative)
        entries = {entry["path"]: entry.get("sha256")
                   for section in sections for entry in document[section]}
        manifest_checks[relative] = {
            "manifest_sha256": sha256(ROOT / relative),
            "bound_entries": {
                path: {"listed": path in entries,
                       "matches_bound_sha256": entries.get(path) == expected}
                for path, expected in binding["source_sha256"].items()
            },
            "qualification": "later_inventory_not_generation_time_dependency_lock",
        }
    batches = [
        inspect_workers(
            ROOT / "artifacts/paper_h3_joint_family_topology_partition/metrics.json",
            [ROOT / "artifacts/paper_h3_joint_family_topology_partition/workers_v5_dev",
             ROOT / "artifacts/paper_h3_joint_family_topology_partition/workers_v5_confirmation"],
        ),
        inspect_workers(
            ROOT / "artifacts/paper_h3_v6_identifiability_tier/metrics.json",
            [ROOT / "artifacts/paper_h3_v6_identifiability_tier/workers_dev_v2"],
        ),
    ]
    main_checks = {}
    readme = None
    if args.main_root:
        for path, expected in binding["source_sha256"].items():
            actual = args.main_root / path
            main_checks[path] = {
                "exists": actual.is_file(),
                "matches_bound_sha256": actual.is_file() and sha256(actual) == expected,
            }
        path = args.main_root / "artifacts/paper_h3_v6_identifiability_tier/README.md"
        if path.is_file():
            readme = {"path": str(path), "sha256": sha256(path), "text": path.read_text(encoding="utf-8-sig")}
    result = {
        "version": "H3_SAVED_PROVENANCE_STATIC_AUDIT_V1",
        "date": "2026-10-09",
        "workspace": str(ROOT),
        "scope": "saved_json_and_exact_source_identity_only",
        "research_runs_started": 0,
        "experiment_modules_imported": False,
        "outputs_written_by_this_tool": False,
        "bound_file_checks": checks,
        "manifest_checks": manifest_checks,
        "audit_tool_sha256": sha256(Path(__file__)),
        "batches": batches,
        "readonly_main_root": str(args.main_root) if args.main_root else None,
        "readonly_main_checks": main_checks,
        "readonly_h3_readme": readme,
        "historical_runtime_dependency_lock": "not_established_by_worker_protocol_hashes",
        "fold_choice_agreement_is_not_proof_of_condition_holdout": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
