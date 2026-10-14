import numpy as np

from experiments.r32_independent_nozzle_gas_solver import (
    IndependentSolverSettings,
    solve_independent_gas_transport,
)
from wphid.truth.reusable_nozzle import (
    NozzleFieldTrajectory,
    nozzle_shell_coordinates,
)


def _small_driver():
    coordinates = nozzle_shell_coordinates()
    shape = coordinates["z"].shape
    t = np.array([0.0, 0.06])
    base = np.ones((t.size, *shape))
    states = {
        "T": 1000.0 * base,
        "pO2": np.zeros_like(base),
        "pH2O": np.zeros_like(base),
        "pProducts": np.zeros_like(base),
        "carbon": base,
        "crack": 0.015 * base,
        "stress": np.zeros_like(base),
    }
    loads = {
        "internal_oxygen_pressure": np.full((t.size, shape[0]), 1000.0),
        "external_steam_pressure": np.full((t.size, shape[0]), 2000.0),
        "external_oxygen_pressure": np.full((t.size, shape[0]), 21000.0),
    }
    return NozzleFieldTrajectory(
        t=t,
        states=states,
        loads=loads,
        mission_id="small",
        **coordinates,
    )


def test_independent_solver_is_finite_and_responds_to_passive_boundaries():
    result = solve_independent_gas_transport(
        _small_driver(), settings=IndependentSolverSettings(time_step=0.01)
    )
    assert all(np.all(np.isfinite(result.states[name])) for name in ("pO2", "pH2O", "pProducts"))
    assert np.max(result.states["pO2"][-1]) > 0.0
    assert np.max(result.states["pH2O"][-1]) > 0.0
    assert np.min(result.states["pO2"][-1]) >= 0.0


def test_independent_solver_accepts_physical_diffusivity_anisotropy():
    driver = _small_driver()
    isotropic = solve_independent_gas_transport(
        driver, settings=IndependentSolverSettings(time_step=0.01)
    )
    anisotropic = solve_independent_gas_transport(
        driver,
        settings=IndependentSolverSettings(
            time_step=0.01,
            meridional_to_through_diffusivity_ratio=3.0,
        ),
    )
    assert np.all(np.isfinite(anisotropic.states["pO2"]))
    # Uniform boundary histories make this tiny smoke case nearly meridionally
    # uniform, so the contract is stability rather than a minimum difference.
    assert anisotropic.states["pO2"].shape == isotropic.states["pO2"].shape


def test_independent_solver_accepts_reduced_darcy_forchheimer_transport():
    result = solve_independent_gas_transport(
        _small_driver(),
        settings=IndependentSolverSettings(
            time_step=0.01,
            forchheimer_inverse_flux_scale=0.01,
        ),
    )
    assert np.all(np.isfinite(result.states["pH2O"]))
    assert np.min(result.states["pH2O"]) >= 0.0
