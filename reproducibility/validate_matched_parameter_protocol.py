"""Read-only checks for the V1 matched-parameter protocol; never runs research.

Exit 0: internally consistent draft. Exit 1: invalid config or reference.
With --require-ready, exit 2: runtime qualification remains unestablished.
Configuration fields cannot grant user authorization or prove implementation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


DEFAULT_PROTOCOL = "docs/research/algorithm-evidence/adam-sindy-matched-comparison-protocol-draft.json"


def binomial_tail(n, k, p):
    return sum(math.comb(n, j) * p**j * (1 - p)**(n - j) for j in range(k, n + 1))


def validate(config, root):
    errors = []
    checks = []
    missing = []

    def check(code, valid):
        (checks if valid else errors).append(code)

    def value(path):
        item = config
        for key in path.split("."):
            if not isinstance(item, dict) or key not in item:
                return None
            item = item[key]
        return item

    check("version", value("version") == "WPHID_ADAM_MATCHED_PARAMETER_PROTOCOL_DRAFT_V1")
    check("scope", value("scope") == "conditional_parameter_comparison_given_family_topology_partition")
    conditions = value("conditions") or {}
    train = value("split.train") or []
    heldout = value("split.heldout") or []
    check("split_names", set(train + heldout) == set(conditions))
    check("split_disjoint", not set(train).intersection(heldout))
    check("split_counts", len(train) == len(set(train)) == 6 and len(heldout) == len(set(heldout)) == 2 and len(conditions) == 8)
    for path in (
        "split.historical_fold_mapping", "split.heldout_used_for_fit",
        "split.heldout_refit_amplitudes", "split.heldout_used_for_restart_selection",
        "model.truth_parameters_exposed_to_estimator", "provenance.writable_main_links_allowed",
        "provenance.old_confirmation_reuse_allowed", "decision.sequential_sampling",
        "decision.planning_probability_is_observed_evidence",
    ):
        check(path, value(path) is False)
    for path in (
        "common_scaling.training_only", "common_scaling.heldout_reuses_training_constants",
        "common_scaling.physical_coefficient_inverse_mapping_required",
        "provenance.complete_runtime_dependency_lock_required",
        "provenance.actual_import_paths_required", "provenance.actual_train_heldout_names_required",
        "provenance.source_and_input_hashes_required",
    ):
        check(path, value(path) is True)
    check("role_bindings", value("model.roles") == [
        {"name": "shared", "known_partition": [0, 0, 0]},
        {"name": "partial", "known_partition": [0, 0, 1]},
        {"name": "independent", "known_partition": [0, 1, 2]},
    ])
    expected_counts = {
        "observations.independent_block_count": 18, "observations.roles_per_block": 3,
        "observations.role_packages": 54, "observations.conditions_per_package": 8,
        "decision.blocks": 18, "decision.required_wins": 13,
        "budget.program_estimates": 108, "budget.max_optimizer_starts": 324,
        "budget.max_initializations_per_program": 3,
        "estimators.wphid.max_initializations": 3,
        "estimators.adam_sindy.max_initializations": 3,
    }
    for path, expected in expected_counts.items():
        check(path, value(path) == expected)
    for path, expected in {
        "decision.alpha": 0.05, "decision.null_max_win_probability": 0.5,
        "decision.planning_win_probability": 0.8, "decision.target_power": 0.8,
        "scoring.relative_improvement_margin": 0.1,
        "scoring.outer_amplitude_relative_error_threshold": 0.25,
    }.items():
        check(path, value(path) == expected)
    check("parameter_thresholds", value("scoring.parameter_relative_error_thresholds") == {
        "theta": 0.12, "oxygen_order": 0.18, "damage_order": 0.18,
    })
    derived = {
        "exact_null_tail": binomial_tail(18, 13, 0.5),
        "exact_power_at_planning_probability": binomial_tail(18, 13, 0.8),
        "zero_errors_per_role_one_sided_95_upper": 1 - 0.05**(1 / 18),
    }
    for key, expected in derived.items():
        actual = value("decision." + key)
        check(key, isinstance(actual, (int, float)) and not isinstance(actual, bool)
              and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
    check("minimum_sample_size", all(
        not any(binomial_tail(n, k, 0.5) <= 0.05 and binomial_tail(n, k, 0.8) >= 0.8
                for k in range(1, n + 1)) for n in range(1, 18)))

    root = Path(root).resolve()
    reference_results = []
    references = value("provenance.legacy_bound_sources") or {}
    check("reference_count", len(references) == 11)
    for name, expected in references.items():
        target = (root / name).resolve()
        safe = not Path(name).is_absolute() and target.is_relative_to(root)
        check("reference_path:" + name, safe)
        if not safe:
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        check("reference_hash:" + name, actual == expected)
        reference_results.append({"path": name, "sha256": actual, "matches": actual == expected})

    for path in (
        "observations.seed_list", "observations.seed_namespace", "observations.input_hashes",
        "estimators.adam_sindy.source_commit_sha", "estimators.adam_sindy.gamma_initialization",
        "estimators.adam_sindy.amplitude_threshold", "budget.shared_cpu_seconds_per_case",
        "budget.objective_evaluation_cap", "budget.backward_evaluation_cap", "budget.hardware",
        "budget.threads", "budget.dtype", "budget.initialization_list", "budget.environment_lock",
        "provenance.output_root",
    ):
        if value(path) is None:
            missing.append(path)
    for estimator in ("wphid", "adam_sindy"):
        if value("estimators." + estimator + ".new_implementation_created") is not True:
            missing.append("estimators." + estimator + ".new_implementation_created")
    seeds = value("observations.seed_list")
    if seeds is not None:
        check("seed_identity", isinstance(seeds, list) and len(seeds) == 18
              and all(type(seed) is int for seed in seeds) and len(set(seeds)) == 18)
        if isinstance(seeds, list):
            check("known_viewed_seeds", all(type(seed) is int and not (
                31900 <= seed <= 31999 or 42100 <= seed <= 42499) for seed in seeds))
    return {
        "audit_kind": "read_only_static_protocol_preflight",
        "structural_checks_passed": not errors, "errors": errors,
        "passed_check_count": len(checks), "missing_configuration_fields": missing,
        "declared_blockers": value("unresolved_blockers"),
        "derived_statistics": derived, "legacy_references": reference_results,
        "runtime_qualification_established": False,
        "qualification_limit": "No estimator behavior, full dependency lock, seed-history exhaustion, input identity, runtime import or cost fairness verified.",
        "execution_permission": "Not determined by config; existing no-experiment user constraint remains in force.",
        "research_runs_started_by_validator": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        raw = (root / args.protocol).read_bytes()
        report = validate(json.loads(raw.decode("utf-8-sig")), root)
        report["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
        report["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["workspace_root"] = str(root)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        report = {"structural_checks_passed": False, "errors": [str(exc)],
                  "research_runs_started_by_validator": 0}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["structural_checks_passed"]:
        return 1
    return 2 if args.require_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
