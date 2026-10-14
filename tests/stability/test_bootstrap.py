import numpy as np
import pytest

from wphid.stability.bootstrap import (
    BootstrappedCondition,
    WeakFormCondition,
    bootstrap_grouped_weak_form,
)


def _conditions():
    return {
        "cold": (np.arange(18.0).reshape(9, 2), np.arange(9.0)),
        "hot": (
            np.arange(100.0, 118.0).reshape(9, 2),
            np.arange(100.0, 109.0),
        ),
    }


def test_grouped_block_bootstrap_is_deterministic_and_keeps_condition_rows_aligned():
    """Seeded draws must be reproducible without mixing condition targets."""
    first = bootstrap_grouped_weak_form(
        _conditions(), block_length=3, random_seed=41
    )
    second = bootstrap_grouped_weak_form(
        _conditions(), block_length=3, random_seed=41
    )

    assert first.condition_ids == second.condition_ids
    for left, right in zip(first.conditions, second.conditions, strict=True):
        assert left.source_condition_id == right.source_condition_id
        np.testing.assert_array_equal(left.row_indices, right.row_indices)
        np.testing.assert_array_equal(left.matrix, right.matrix)
        np.testing.assert_array_equal(left.target, right.target)
        source_matrix, source_target = _conditions()[left.source_condition_id]
        np.testing.assert_array_equal(left.matrix, source_matrix[left.row_indices])
        np.testing.assert_array_equal(left.target, source_target[left.row_indices])


def test_grouped_block_bootstrap_resamples_complete_conditions_and_contiguous_blocks():
    """Each draw is a whole condition whose windows arrive in bounded runs."""
    sample = bootstrap_grouped_weak_form(
        _conditions(), block_length=3, random_seed=5
    )

    assert len(sample.conditions) == 2
    assert {item.source_condition_id for item in sample.conditions} <= {"cold", "hot"}
    for item in sample.conditions:
        assert item.matrix.shape == (9, 2)
        assert item.target.shape == (9,)
        assert np.all((item.row_indices >= 0) & (item.row_indices < 9))
        offset = 0
        for length in item.block_lengths:
            indices = item.row_indices[offset : offset + length]
            np.testing.assert_array_equal(
                indices, np.arange(indices[0], indices[0] + length)
            )
            offset += length
        assert offset == item.row_indices.size


def test_grouped_block_bootstrap_rejects_unaligned_or_incompatible_conditions():
    with pytest.raises(ValueError, match="same number of columns"):
        bootstrap_grouped_weak_form(
            {
                "a": (np.ones((4, 2)), np.ones(4)),
                "b": (np.ones((4, 3)), np.ones(4)),
            },
            block_length=2,
            random_seed=0,
        )

    with pytest.raises(ValueError, match="same number of rows"):
        WeakFormCondition("a", np.ones((4, 2)), np.ones(3))

def test_grouped_block_bootstrap_rejects_single_row_blocks():
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_grouped_weak_form(
            _conditions(), block_length=1, random_seed=0
        )


def test_nondivisible_condition_is_padded_with_complete_contiguous_blocks():
    sample = bootstrap_grouped_weak_form(
        {"only": _conditions()["cold"]},
        block_length=4,
        random_seed=11,
    )

    condition = sample.conditions[0]
    assert condition.block_lengths == (4, 4, 4)
    assert condition.row_indices.size == 12
    for offset in range(0, 12, 4):
        block = condition.row_indices[offset : offset + 4]
        np.testing.assert_array_equal(
            block, np.arange(block[0], block[0] + 4)
        )


def test_bootstrapped_condition_rejects_noninteger_row_indices():
    with pytest.raises(ValueError, match="integer"):
        BootstrappedCondition(
            condition_id="bootstrap_0",
            source_condition_id="source",
            matrix=np.ones((2, 1)),
            target=np.ones(2),
            row_indices=np.array([0.5, 1.5]),
            block_lengths=(2,),
        )