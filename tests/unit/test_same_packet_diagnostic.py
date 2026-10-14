from types import SimpleNamespace

import numpy as np
import pytest

from experiments.audit_same_packet_weak_forward import (
    FIELDS, finite_volume_balance, forward_score, frozen_observations,
)
from wphid.transport_mechanisms import TransportCandidate
from wphid.truth.reusable_nozzle import NozzleGeometry, NozzleGrid, _shell_mesh, _shell_diffusion


def fixture():
    mesh = _shell_mesh(NozzleGeometry(), NozzleGrid(meridional_cells=31, thickness_cells=8))
    t = np.linspace(0, 180, 31)
    fields = {n: np.zeros((31,31,8)) for n in FIELDS}
    fields["T"][:] = 1000.
    fields["carbon"][:] = 1.
    trajectory = SimpleNamespace(t=t, states=fields, **{n:mesh[n] for n in ("s","x","z","r")})
    return mesh, trajectory


def test_packet_is_detached_readonly_and_hash_changes_with_observations():
    _, trajectory = fixture()
    packet, identity = frozen_observations(trajectory)
    trajectory.states["T"][0,0,0] += 1.
    assert packet["T"][0,0,0] == 1000.
    assert frozen_observations(trajectory)[1] != identity
    with pytest.raises(ValueError):
        packet["T"][0,0,0] = 1.


def test_nonfinite_observation_is_rejected():
    _, trajectory = fixture()
    trajectory.states["T"][0,0,0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        frozen_observations(trajectory)


def test_zero_gas_balance_and_self_template_score_are_zero():
    mesh, trajectory = fixture()
    packet, _ = frozen_observations(trajectory)
    assert forward_score(packet,trajectory)[0] == 0.
    for law, alpha in (("Darcy",0.),("Forchheimer",2.)):
        result = finite_volume_balance(packet,TransportCandidate("combined_opening",law,3.,alpha),
            "pO2",np.zeros_like(packet["pO2"]),mesh)
        assert result["rows"] > 0
        assert result["residual_rms"] == 0.


def test_discrete_diffusion_conserves_volume_integral_without_boundary_sources():
    mesh, _ = fixture()
    field = np.arange(31*8,dtype=float).reshape(31,8)**2
    coefficient = np.full_like(field,4.e-7)
    for alpha in (0.,2.):
        tendency = _shell_diffusion(field,3*coefficient,coefficient,mesh,
            forchheimer_inverse_flux_scale=alpha)
        integrated = tendency*mesh["volume"]
        assert abs(float(np.sum(integrated))) < 1.e-12*float(np.sum(np.abs(integrated)))
