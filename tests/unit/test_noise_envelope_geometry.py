from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

spec = spec_from_file_location("envelope_audit", Path(__file__).resolve().parents[2] / "experiments/audit_noise_envelope_geometry.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_ellipsoid_volume_for_diagonal_two_dimensions():
    method = dict(infinite_radius=False, calibrated_squared_radius=4.,
        metric=dict(scale=[2., 3.], correlation_eigenvalues=[1., 1.], eigenvalue_floor=1.e-6))
    assert module.ellipsoid_log_volume_without_unit_ball(method) == pytest.approx(np.log(24.))


def test_infinite_radius_is_not_a_finite_volume():
    with pytest.raises(ValueError):
        module.ellipsoid_log_volume_without_unit_ball(dict(infinite_radius=True,
            calibrated_squared_radius=None, metric={}))
