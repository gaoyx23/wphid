import numpy as np

from wphid.eiv.network import fit_network_simex_path


def test_network_simex_preserves_a_constant_exact_path():
    x = np.linspace(550.0, 2200.0, 100)

    def reconstruction(level, replicate):
        del level, replicate
        bases = [np.ones((x.size, 1)), np.ones((x.size, 1))]
        targets = [0.2+2*np.exp(-7000/x), -0.1-np.exp(-7000/x)]
        return bases, targets, lambda p: [np.exp(-p[0]/x)]*2

    result = fit_network_simex_path(
        [0.0, 0.5, 1.0], reconstruction, [True, True],
        [(2000.0, 14000.0)], [(-10.0, 10.0)]*4,
        n_replicates=2, seed=9, global_iterations=20)
    assert result.stable
    assert np.allclose(result.extrapolated_parameters, [7000.0], rtol=2e-4)
    assert np.allclose(result.extrapolated_coefficients, [.2, 2., -.1, -1.], rtol=2e-4)
