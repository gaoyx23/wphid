"""Actual bounded-update and unit-conversion contracts on artificial inputs."""

from types import SimpleNamespace
import numpy as np
import pytest

from wphid.precision_update_v4.update import one_step, physical_units


class Risk:
    def __init__(self):
        self.calls = []

    def profile(self, values, **options):
        self.calls.append((values.copy(), options))
        return SimpleNamespace(objective=-.4, coefficients=(np.ones(4),)*3)


def test_step_is_applied_once_in_physical_coordinates():
    risk = Risk()
    proposed, result = one_step(risk, [6000., .5, 1.2], [[2000., 14000.], [0., 2.], [0., 3.]], [.01, -.02, .015])
    np.testing.assert_allclose(proposed, [6120., .46, 1.245])
    assert len(risk.calls) == 1
    assert risk.calls[0][1] == {'mean_correction': True, 'gram_correction': True}
    assert result.objective == -.4


def test_zero_step_is_not_turned_into_a_new_search():
    risk = Risk()
    proposed, _ = one_step(risk, [3., 4.], [[0., 10.], [0., 10.]], [0., 0.])
    np.testing.assert_array_equal(proposed, [3., 4.])
    assert len(risk.calls) == 1


@pytest.mark.parametrize('start,step,error', [(5., .11, 'radius'), (9.8, .04, 'physical box')])
def test_invalid_step_is_not_clipped_or_damped(start, step, error):
    risk = Risk()
    with pytest.raises(ValueError, match=error):
        one_step(risk, [start], [[0., 10.]], [step])
    assert not risk.calls


def test_physical_units_do_not_apply_target_weight_twice():
    scale = SimpleNamespace(base_norms=(np.array([2., 4., 8.]),)*3,
                            norms_for=lambda family: (.5, .25, .125))
    values = (np.array([2., 8., 24., 4.]),)*3
    physical = physical_units(scale, None, values)
    np.testing.assert_allclose(physical[0], [1., 2., 3., 8.])
    np.testing.assert_allclose(physical[2], [1., 2., 3., 32.])


def test_nonfinite_step_is_rejected():
    with pytest.raises(ValueError, match='layout'):
        one_step(Risk(), [5.], [[0., 10.]], [np.nan])
