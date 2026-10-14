import inspect

import numpy as np

from experiments import paper_h1_heterogeneous_three_class_network as experiment


def test_protocol_freezes_roles_splits_metrics_and_confirmation_gates():
    protocol = experiment.PROTOCOL
    assert protocol["confirmation_locked"] is False
    assert set(protocol["roles"]) == {
        "full", "without_intrinsic", "without_directed", "without_shared", "null"
    }
    assert all(set(scales) == set(experiment.MECHANISM_CLASSES)
               for scales in protocol["roles"].values())
    assert set(protocol["development_seeds"]).isdisjoint(protocol["confirmation_seeds"])
    assert protocol["minimum_confirmation_rate_per_role"] == 0.8
    assert protocol["maximum_null_false_acceptance_rate"] == 0.2
    assert set(protocol["reported_metrics"]) >= {
        "family_exact_by_class", "topology_exact_by_class", "overall_exact",
        "bic_margin", "parameter_relative_error",
        "mechanism_coefficient_relative_error", "truth_candidate_screen_rank",
        "truth_candidate_screen_admitted", "null_false_acceptance",
    }


def test_source_signatures_are_disjoint_and_candidate_space_contains_truth():
    expected_signatures = {
        "intrinsic": {(False, False, True)},
        "directed": {(False, True, True)},
    }
    for mechanism, expected in expected_signatures.items():
        observed = {tuple(specification["source_signature"])
                    for specification in experiment.FAMILIES[mechanism].values()}
        assert observed == expected
    assert all(specification["source_signature"][0]
               for specification in experiment.FAMILIES["shared"].values())
    assert "damage_power" not in experiment.FAMILIES["directed"]
    assert "oxygen_remaining_power" in experiment.FAMILIES["directed"]
    for mechanism in experiment.MECHANISM_CLASSES:
        truth = experiment.TRUTH[mechanism]
        assert truth["family"] in experiment.FAMILIES[mechanism]
        options = list(experiment._mechanism_options(mechanism))
        assert any(option["family"] is None for option in options)
        for family in experiment.FAMILIES[mechanism]:
            assert {tuple(option["topology"]) for option in options
                    if option["family"] == family} == set(experiment.OPEN_TOPOLOGIES)
    assert experiment.PROTOCOL["joint_candidate_count"] >= 10_000


def test_joint_selector_stratifies_all_presence_patterns_without_truth_input():
    signature = inspect.signature(experiment._select_stratified_candidates)
    assert tuple(signature.parameters) == ("rows", "pattern_quotas")
    assert "truth" not in inspect.getsource(
        experiment._select_stratified_candidates
    ).lower()
    rows = []
    for pattern in experiment.PRESENCE_PATTERNS:
        for index in range(3):
            rows.append({"name": f"{pattern}-{index}",
                         "presence_pattern": list(pattern),
                         "screen_bic": float(3-index)})
    selected = experiment._select_stratified_candidates(
        rows, experiment.PROTOCOL["presence_pattern_quotas"]
    )
    for pattern in experiment.PRESENCE_PATTERNS:
        expected = experiment.PROTOCOL["presence_pattern_quotas"][
            experiment._pattern_name(pattern)
        ]
        assert sum(tuple(row["presence_pattern"]) == pattern
                   for row in selected) == expected


def test_role_truth_scoring_switches_absent_classes_to_null():
    for scales in experiment.ROLES.values():
        expected = experiment._truth_selections(scales)
        for mechanism in experiment.MECHANISM_CLASSES:
            if scales[mechanism] == 0.0:
                assert expected[mechanism] == {
                    "family": None, "topology": list(experiment.NULL_TOPOLOGY)
                }
            else:
                assert expected[mechanism]["family"] == experiment.TRUTH[mechanism]["family"]
                assert expected[mechanism]["topology"] == experiment.TRUTH[mechanism]["topology"]
    assert all(value == 0.0 for value in experiment.ROLES["null"].values())


def test_small_manufactured_role_uses_axisymmetric_weak_pipeline():
    local, bases, targets = experiment._assemble(
        experiment.PROTOCOL["development_seeds"][0], experiment.ROLES["full"],
        condition_names=(next(iter(experiment.CONDITIONS)),), noise_fraction=0.0,
    )
    assert len(local) == 1
    assert len(bases) == len(targets) == len(experiment.FIELDS) == 3
    assert all(base.shape[0] == target.size and base.shape[1] == 3
               for base, target in zip(bases, targets))
    assert all(np.all(np.isfinite(base)) and np.all(np.isfinite(target))
               for base, target in zip(bases, targets))


def test_worker_rejects_seed_outside_declared_split(tmp_path):
    try:
        experiment.run_worker(99999, "full", "development", tmp_path/"row.json")
    except ValueError as error:
        assert "frozen split" in str(error)
    else:
        raise AssertionError("undeclared seed was accepted")
