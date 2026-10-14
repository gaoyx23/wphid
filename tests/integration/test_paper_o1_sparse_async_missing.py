import numpy as np

from experiments.paper_o1_sparse_async_missing import (
    OBSERVATION_SPEC,
    PROTOCOL,
    reconstruct_field,
)


def test_observation_protocol_is_sparse_asynchronous_and_missing():
    assert len({spec["time_samples"] for spec in OBSERVATION_SPEC.values()}) == 3
    assert all(spec["axial_sensors"] < 25 and spec["radial_sensors"] < 25
               for spec in OBSERVATION_SPEC.values())
    assert all(spec["missing_fraction"] > 0 for spec in OBSERVATION_SPEC.values())
    assert PROTOCOL["minimum_role_success_rate"] == 0.8


def test_reconstruction_preserves_an_affine_field_without_noise_or_missingness():
    t = np.linspace(0.0, 1.0, 25)
    z = np.linspace(-0.5, 0.8, 25)
    r = np.linspace(0.0, 1.0, 25)
    tt, zz, rr = np.meshgrid(t, z, r, indexing="ij")
    truth = 3.0 + 0.2 * tt - 0.4 * zz + 0.7 * rr
    spec = {"axial_sensors": 7, "radial_sensors": 7,
            "time_samples": 9, "missing_fraction": 0.0}
    recovered, diagnostics = reconstruct_field(
        t, z, r, truth, spec, np.random.default_rng(1),
        noise_fraction=0.0, jitter_fraction=0.2,
    )
    assert np.allclose(recovered, truth, atol=1e-12)
    assert diagnostics["spatial_sensor_fraction"] < 0.1
