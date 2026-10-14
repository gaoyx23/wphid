import numpy as np

import experiments.s3c_e4_frustum_autonomous_forecast as experiment


def test_mapped_frustum_operator_annihilates_constant():
    geometry = experiment.GEOMETRIES["thin_tapered"]
    z = np.linspace(0., geometry.length, 9)
    eta = np.linspace(0., 1., 13)
    axial, radial = experiment.mapped_frustum_operators(
        np.ones((z.size, eta.size)), z, eta, geometry)
    assert np.allclose(axial, 0.)
    assert np.allclose(radial, 0.)


def test_s3c_e4_single_closed_loop_case():
    row = experiment.run_case("nominal", 34401)
    assert set(row["field_nrmse"]) == set(experiment.FIELDS)
