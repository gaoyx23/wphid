import numpy as np

from wphid.stability.selection import run_stability_selection


def test_run_stability_selection_uses_only_bootstrapped_training_conditions():
    conditions = {
        "train-a": (np.column_stack((np.arange(8.0), np.ones(8))), np.arange(8.0)),
        "train-b": (np.column_stack((np.arange(20.0, 28.0), np.ones(8))), np.arange(20.0, 28.0)),
    }
    observed_sources = []

    def selector(matrix, target, random_seed, rng):
        observed_sources.append((matrix.copy(), target.copy()))
        return np.array([1.0, 0.0]), 1800.0

    summary = run_stability_selection(
        conditions,
        candidate_names=("signal", "noise"),
        selector=selector,
        n_resamples=5,
        block_length=2,
        random_seed=9,
    )

    assert len(observed_sources) == 5
    assert summary.n_resamples == 5
    assert dict(summary.parameter_selection_frequencies) == {1800.0: 1.0}
    np.testing.assert_allclose(summary.inclusion_probabilities, [1.0, 0.0])

def test_run_stability_selection_repeats_random_selector_deterministically():
    conditions = {
        "train": (
            np.column_stack((np.arange(9.0), np.ones(9))),
            np.arange(9.0),
        )
    }

    def run_once():
        observations = []

        def selector(matrix, target, random_seed, rng):
            draw = rng.normal(size=2)
            observations.append((random_seed, draw.copy()))
            return draw, None

        summary = run_stability_selection(
            conditions,
            candidate_names=("a", "b"),
            selector=selector,
            n_resamples=6,
            block_length=4,
            random_seed=73,
        )
        return summary, observations

    first, first_observations = run_once()
    second, second_observations = run_once()

    assert len({seed for seed, _ in first_observations}) == 6
    assert [seed for seed, _ in first_observations] == [
        seed for seed, _ in second_observations
    ]
    for (_, first_draw), (_, second_draw) in zip(
        first_observations, second_observations, strict=True
    ):
        np.testing.assert_array_equal(first_draw, second_draw)
    np.testing.assert_array_equal(
        first.inclusion_probabilities, second.inclusion_probabilities
    )
    np.testing.assert_array_equal(first.sign_consistency, second.sign_consistency)


def test_validation_and_test_sentinels_never_enter_selector():
    all_conditions = {
        "train-a": (np.ones((6, 1)), np.arange(6.0)),
        "train-b": (np.ones((6, 1)), np.arange(10.0, 16.0)),
        "validation-sentinel": (np.ones((6, 1)), np.full(6, 10_001.0)),
        "test-sentinel": (np.ones((6, 1)), np.full(6, 20_001.0)),
    }
    training_conditions = {
        name: all_conditions[name] for name in ("train-a", "train-b")
    }
    observed_targets = []

    def selector(matrix, target, random_seed, rng):
        observed_targets.append(target.copy())
        return np.array([1.0])

    run_stability_selection(
        training_conditions,
        candidate_names=("factor",),
        selector=selector,
        n_resamples=5,
        block_length=2,
        random_seed=19,
    )

    assert len(observed_targets) == 5
    assert all(np.max(target) < 10_000.0 for target in observed_targets)