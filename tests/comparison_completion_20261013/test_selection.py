"""Selection must keep unassessable cases and handle both directions equally."""

from reproducibility.comparison_completion_20261013.collect_results import directional_blocks


def test_failures_and_zero_comparator_scores_cannot_be_wins():
    assessment = {'blocks': [
        {'seed': 1, 'scores': {'wphid': .1, 'adam': None}, 'both_assessable': False},
        {'seed': 2, 'scores': {'wphid': 0., 'adam': 0.}, 'both_assessable': True},
        {'seed': 3, 'scores': {'wphid': 1., 'adam': 1.}, 'both_assessable': True},
    ]}
    for direction in ('wphid', 'adam'):
        rows = directional_blocks(assessment, direction)
        assert len(rows) == 3
        assert not any(row['win'] for row in rows)


def test_same_effect_threshold_for_both_directions():
    assessment = {'blocks': [
        {'seed': 1, 'scores': {'wphid': .89, 'adam': 1.}, 'both_assessable': True},
        {'seed': 2, 'scores': {'wphid': 1., 'adam': .89}, 'both_assessable': True},
        {'seed': 3, 'scores': {'wphid': .91, 'adam': 1.}, 'both_assessable': True},
        {'seed': 4, 'scores': {'wphid': 1., 'adam': .91}, 'both_assessable': True},
    ]}
    assert [r['win'] for r in directional_blocks(assessment, 'wphid')] == [True, False, False, False]
    assert [r['win'] for r in directional_blocks(assessment, 'adam')] == [False, True, False, False]
