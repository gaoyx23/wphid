from __future__ import annotations

import numpy as np

from experiments.paper_r100_geometry_material_defect_transfer import (
    GRID,
    initial_defect_fields,
    transfer_cases,
)


def test_transfer_cases_isolate_axes_and_include_combined_case():
    cases = transfer_cases()
    reference = cases["reference_bartz"]
    geometry = cases["geometry_envelope_shift"]
    material = cases["material_batch_shift"]
    defect = cases["initial_defect_shift"]
    combined = cases["combined_shift"]

    assert geometry.geometry != reference.geometry
    assert geometry.material == reference.material
    assert material.geometry == reference.geometry
    assert material.material != reference.material
    assert defect.heterogeneous_initial_defect
    assert combined.geometry == geometry.geometry
    assert combined.material == material.material
    assert combined.heterogeneous_initial_defect

    for case in cases.values():
        z = np.linspace(0.0, case.geometry.length, 257)
        assert np.all(np.diff(case.geometry.inner_radius(z)) > 0.0)
        assert 20.0 < case.geometry.shell_mass(case.material.density) < 30.0


def test_initial_defect_fields_are_bounded_smooth_and_nonuniform():
    carbon, crack = initial_defect_fields(GRID)
    assert carbon.shape == (GRID.meridional_cells, GRID.thickness_cells)
    assert crack.shape == carbon.shape
    assert 0.94 <= float(carbon.min()) < float(carbon.max()) <= 1.0
    assert 0.005 <= float(crack.min()) < float(crack.max()) <= 0.08
    assert float(np.max(np.abs(np.diff(carbon, axis=0)))) < 0.02
    assert float(np.max(np.abs(np.diff(crack, axis=0)))) < 0.02
