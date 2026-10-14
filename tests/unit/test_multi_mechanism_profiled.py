import numpy as np
import pytest

from wphid.discovery.network_profiled import (
    profiled_multi_mechanism_fit,
    profiled_partitioned_mechanism_fit,
)


def test_two_shared_mechanisms_recover_separate_parameters_and_topologies():
    x = np.linspace(0.15, 1.2, 180)
    z = np.linspace(0.2, 1.1, 180) ** 1.3
    bases = (np.column_stack((np.ones(x.size), x)),
             np.column_stack((np.ones(x.size), z)))

    def first(parameters):
        atom = np.exp(-parameters[0] * x)
        return atom, atom

    def second(parameters):
        atom = z ** parameters[0]
        return atom, atom

    first_truth = np.exp(-0.7 * x)
    second_truth = z ** 1.4
    targets = (bases[0] @ np.array([0.3, -0.2]) + 1.8 * first_truth
               + 0.55 * second_truth,
               bases[1] @ np.array([-0.1, 0.25]) - 0.8 * first_truth)
    fit = profiled_multi_mechanism_fit(
        bases, targets, (first, second), ((True, True), (True, False)),
        (((0.2, 1.2),), ((0.6, 2.2),)), seed=41, global_iterations=30,
    )
    assert np.allclose([fit.parameters[0][0], fit.parameters[1][0]],
                       [0.7, 1.4], rtol=2e-3, atol=2e-3)
    assert fit.active_equations == ((True, True), (True, False))
    assert np.allclose(fit.coefficients[0][-2:], [1.8, 0.55], rtol=2e-3)
    assert np.isclose(fit.coefficients[1][-1], -0.8, rtol=2e-3)


def test_multi_mechanism_fit_accepts_a_local_warm_start():
    x = np.linspace(0.1, 1.0, 180)
    bases = (np.ones((x.size, 1)),)
    targets = (0.2 + 1.7 * x**1.2 - 0.8 * np.exp(-0.9 * x),)
    fit = profiled_multi_mechanism_fit(
        bases, targets,
        (lambda p: (x**p[0],), lambda p: (np.exp(-p[0] * x),)),
        ((True,), (True,)), (((0.5, 2.0),), ((0.2, 1.8),)),
        initial_parameters=(1.15, 0.85), use_global_search=False,
    )
    assert fit.parameters[0][0] == pytest.approx(1.2, rel=2e-3)
    assert fit.parameters[1][0] == pytest.approx(0.9, rel=2e-3)


def _sharing_control_problem(parameters, seed=73):
    rng = np.random.default_rng(seed)
    temperature = np.linspace(650.0, 2100.0, 260)
    bases = (np.column_stack((np.ones(temperature.size), temperature / 2100.0)),
             np.column_stack((np.ones(temperature.size), (temperature / 2100.0) ** 2)))

    def builder(values):
        atom = np.exp(-values[0] / temperature)
        return atom, atom

    targets = []
    for equation, (base, theta, amplitude) in enumerate(
            zip(bases, parameters, (2.3, -1.7))):
        clean = base @ np.array([0.15, -0.08]) + amplitude * np.exp(-theta / temperature)
        targets.append(clean + rng.normal(0.0, 2e-4, clean.shape))
    return bases, tuple(targets), builder


def test_partitioned_fit_prefers_true_shared_parameter_group():
    bases, targets, builder = _sharing_control_problem((7200.0, 7200.0))
    shared = profiled_partitioned_mechanism_fit(
        bases, targets, (builder,), ((0, 0),), (((2500.0, 13000.0),),),
        seed=17, global_iterations=25,
    )
    independent = profiled_partitioned_mechanism_fit(
        bases, targets, (builder,), ((0, 1),), (((2500.0, 13000.0),),),
        seed=17, global_iterations=25,
    )
    assert shared.bic < independent.bic
    assert shared.parameter_groups == ((0, 0),)
    assert shared.parameters[0][0][0] == pytest.approx(7200.0, rel=0.03)


def test_partitioned_fit_rejects_false_sharing():
    bases, targets, builder = _sharing_control_problem((5200.0, 9800.0))
    shared = profiled_partitioned_mechanism_fit(
        bases, targets, (builder,), ((0, 0),), (((2500.0, 13000.0),),),
        seed=23, global_iterations=25,
    )
    independent = profiled_partitioned_mechanism_fit(
        bases, targets, (builder,), ((0, 1),), (((2500.0, 13000.0),),),
        seed=23, global_iterations=25,
    )
    assert independent.bic < shared.bic
    recovered = [group[0] for group in independent.parameters[0]]
    assert np.allclose(recovered, [5200.0, 9800.0], rtol=0.04)


def test_partitioned_profile_has_finite_bic_for_exact_closed_target():
    x = np.linspace(0.2, 1.0, 40)
    bases = (np.ones((x.size, 1)), np.ones((x.size, 1)))

    def builder(parameters):
        atom = x ** parameters[0]
        return atom, atom

    targets = (2.0 + 3.0 * x ** 1.2, -1.0 + 0.5 * x ** 1.2)
    fit = profiled_partitioned_mechanism_fit(
        bases, targets, (builder,), ((0, 0),), (((0.5, 2.0),),),
        seed=7, global_iterations=8,
    )
    assert np.isfinite(fit.bic)
    assert abs(fit.parameters[0][0][0] - 1.2) < 1e-4


def test_partitioned_local_warm_start_matches_exact_shared_solution():
    x = np.linspace(0.2, 1.0, 40)
    bases = (np.ones((x.size, 1)), np.ones((x.size, 1)))

    def builder(parameters):
        atom = x ** parameters[0]
        return atom, atom

    targets = (2.0 + 3.0 * x ** 1.2, -1.0 + 0.5 * x ** 1.2)
    fit = profiled_partitioned_mechanism_fit(
        bases, targets, (builder,), ((0, 0),), (((0.5, 2.0),),),
        initial_parameters=(1.15,), use_global_search=False,
    )
    assert abs(fit.parameters[0][0][0] - 1.2) < 1e-4


def test_partitioned_local_fit_requires_warm_start():
    bases, targets, builder = _sharing_control_problem((7200.0, 7200.0))
    with pytest.raises(ValueError, match="requires initial parameters"):
        profiled_partitioned_mechanism_fit(
            bases, targets, (builder,), ((0, 0),),
            (((2500.0, 13000.0),),), use_global_search=False,
        )


def test_separable_group_fit_recovers_independent_h3_parameters():
    from experiments import paper_h3_joint_family_topology_partition as h3

    local, bases, targets = h3._assemble(
        43101, "independent", noise_fraction=0.0,
        discrete_consistent_target=True,
    )
    fit = h3._fit_partition(
        43101, local, bases, targets, h3.TRUTH_FAMILY, h3.TRUTH_TOPOLOGY,
        (0, 1, 2), iterations=8, separable_groups=True,
    )
    recovered = np.asarray([group for group in fit.parameters[0]])
    truth = np.asarray(h3.TRUTH_CASES["independent"]["parameters"])
    assert np.allclose(recovered, truth, rtol=1e-5, atol=1e-5)
