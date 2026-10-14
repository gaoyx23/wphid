import numpy as np
import pytest

from wphid.discovery.network_profiled import profiled_shared_mechanism_fit


def test_shared_parameter_is_recovered_once_with_opposite_outer_signs():
    temperature = np.linspace(550.0, 2200.0, 240)
    truth = 7200.0
    atom = np.exp(-truth/temperature)
    bases = [np.ones((temperature.size, 1)), np.ones((temperature.size, 1))]
    targets = [0.2+3.0*atom, -0.1-1.5*atom]

    fit = profiled_shared_mechanism_fit(
        bases, targets,
        lambda parameters: [np.exp(-parameters[0]/temperature)]*2,
        [True, True], [(2000.0, 14000.0)], seed=17,
        global_iterations=30,
    )

    assert fit.parameters[0] == pytest.approx(truth, rel=2e-4)
    assert fit.coefficients[0][-1] == pytest.approx(3.0, rel=2e-4)
    assert fit.coefficients[1][-1] == pytest.approx(-1.5, rel=2e-4)
    assert fit.active_equations == (True, True)


def test_shared_mechanism_requires_an_active_equation():
    with pytest.raises(ValueError, match="at least one"):
        profiled_shared_mechanism_fit(
            [np.ones((5, 1))], [np.ones(5)], lambda _: [np.ones(5)],
            [False], [(1.0, 2.0)],
        )
