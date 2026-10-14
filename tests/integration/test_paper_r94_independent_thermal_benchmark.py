import numpy as np

from experiments.paper_r94_independent_thermal_engineering_benchmark import (
    _bartz_normalized_shape,
    _solve_independent_columns,
)
from wphid.truth.reusable_nozzle import NozzleGeometry, NozzleMission
from wphid.truth.reusable_tps import TPSMaterial


def test_bartz_type_shape_is_throat_peaked_and_monotone():
    geometry = NozzleGeometry()
    z = np.linspace(0.0, geometry.length, 31)
    profile = _bartz_normalized_shape(z, geometry, NozzleMission())

    np.testing.assert_allclose(profile["shape"][0], 1.0)
    assert np.all(np.diff(profile["shape"]) < 0.0)
    assert profile["shape"][-1] < 0.05
    assert np.all(profile["mach"] >= 1.0)


def test_independent_wall_solver_stays_isothermal_without_load_or_radiation_drive():
    material = TPSMaterial(emissivity=0.0)
    times = np.linspace(0.0, 20.0, 11)
    flux = np.zeros((times.size, 3))
    temperature = _solve_independent_columns(
        times,
        flux,
        thickness=0.0065,
        cells=6,
        material=material,
        ambient_temperature=330.0,
        initial_temperature=330.0,
    )

    np.testing.assert_allclose(temperature, 330.0, atol=1e-8)
