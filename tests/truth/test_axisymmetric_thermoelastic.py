import numpy as np
import pytest

from wphid.truth.axisymmetric_thermoelastic import (
    lame_thick_cylinder_stress,
    solve_axisymmetric_ring,
)


def test_pressure_only_solution_matches_lame_thick_cylinder():
    inner_radius = 0.05
    outer_radius = 0.06
    element_count = 64
    depth = (np.arange(element_count) + 0.5) * (
        (outer_radius - inner_radius) / element_count
    )
    result = solve_axisymmetric_ring(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        source_depth=np.array([0.0, outer_radius - inner_radius]),
        temperature=np.array([330.0, 330.0]),
        young_modulus=70.0e9,
        thermal_expansion=2.0e-6,
        poisson_ratio=0.2,
        reference_temperature=330.0,
        internal_pressure=2.0e6,
        element_count=element_count,
        sample_depth=depth,
    )
    exact_radial, exact_hoop = lame_thick_cylinder_stress(
        inner_radius + depth,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        internal_pressure=2.0e6,
    )

    assert np.linalg.norm(result.radial_stress - exact_radial) / np.linalg.norm(
        exact_radial
    ) < 2.0e-5
    assert np.linalg.norm(result.hoop_stress - exact_hoop) / np.linalg.norm(
        exact_hoop
    ) < 2.0e-5


def test_uniform_free_heating_expands_without_spurious_stress():
    result = solve_axisymmetric_ring(
        inner_radius=0.05,
        outer_radius=0.06,
        source_depth=np.array([0.0, 0.01]),
        temperature=np.array([1000.0, 1000.0]),
        young_modulus=70.0e9,
        thermal_expansion=2.0e-6,
        poisson_ratio=0.2,
        reference_temperature=330.0,
        internal_pressure=0.0,
        closed_end_pressure=False,
        element_count=48,
        sample_depth=np.array([0.002, 0.005, 0.008]),
    )

    assert np.max(result.von_mises_stress) < 1.0
    expected = 2.0e-6 * (1000.0 - 330.0) * (0.05 + result.sample_depth)
    np.testing.assert_allclose(result.radial_displacement, expected, rtol=2.0e-10)


def test_thermal_gradient_is_mesh_converged_at_nozzle_wall_samples():
    depth = (np.arange(8) + 0.5) * (0.0065 / 8.0)
    common = {
        "inner_radius": 0.06,
        "outer_radius": 0.0665,
        "source_depth": depth,
        "temperature": np.linspace(1800.0, 700.0, 8),
        "young_modulus": np.linspace(58.0e9, 60.0e9, 8),
        "thermal_expansion": np.linspace(2.4e-6, 1.1e-6, 8),
        "poisson_ratio": 0.032,
        "reference_temperature": 330.0,
        "internal_pressure": 0.9e6,
        "axial_restraint_fraction": 0.28,
        "sample_depth": depth,
    }
    coarse = solve_axisymmetric_ring(**common, element_count=64)
    fine = solve_axisymmetric_ring(**common, element_count=128)
    relative = np.linalg.norm(
        coarse.von_mises_stress - fine.von_mises_stress
    ) / np.linalg.norm(fine.von_mises_stress)

    assert relative < 0.01
    assert not fine.von_mises_stress.flags.writeable


@pytest.mark.parametrize(
    "override",
    (
        {"inner_radius": -0.1},
        {"outer_radius": 0.04},
        {"poisson_ratio": 0.5},
        {"internal_pressure": -1.0},
        {"axial_restraint_fraction": 1.1},
        {"element_count": 1},
    ),
)
def test_invalid_ring_inputs_are_rejected(override):
    arguments = {
        "inner_radius": 0.05,
        "outer_radius": 0.06,
        "source_depth": np.array([0.0, 0.01]),
        "temperature": np.array([330.0, 330.0]),
        "young_modulus": 70.0e9,
        "thermal_expansion": 2.0e-6,
        "poisson_ratio": 0.2,
        "reference_temperature": 330.0,
        "internal_pressure": 2.0e6,
    }
    arguments.update(override)
    with pytest.raises(ValueError):
        solve_axisymmetric_ring(**arguments)
