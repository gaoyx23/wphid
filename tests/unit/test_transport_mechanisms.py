from dataclasses import replace
import numpy as np
import pytest

from wphid.transport_mechanisms import MODULATIONS, TransportCandidate, modulation_fraction, constitutive_flux
from wphid.truth.reusable_cmc import CMCMechanics
from wphid.truth.reusable_nozzle import (
    NozzleGrid, NozzleMission, NozzleTransport, _crack_opening,
    simulate_reusable_nozzle_mission,
)


def states():
    return dict(temperature=np.array([300., 1600., 2000.]),
                stress=np.array([0., 38.e6, 114.e6]), carbon=np.array([1., .9, .5]),
                crack=np.array([.01, .2, .7]), transport=NozzleTransport(), mechanics=CMCMechanics())


def test_combined_modulation_exactly_matches_existing_formula():
    s = states()
    legacy = np.clip((_crack_opening(s["temperature"], s["stress"], s["carbon"],
        s["crack"], s["transport"], s["mechanics"]) - 1.)/9.6, 0., 1.)
    np.testing.assert_array_equal(modulation_fraction("combined_opening", **s), legacy)


def test_nonlinear_flux_satisfies_physical_relation_and_preserves_sign():
    linear = np.array([-5., -1., 0., 1., 5.])
    flux = constitutive_flux(linear, 2.)
    np.testing.assert_allclose(flux + 2.*np.abs(flux)*flux, linear)
    np.testing.assert_array_equal(np.sign(flux), np.sign(linear))
    np.testing.assert_array_equal(constitutive_flux(linear, 0.), linear)


@pytest.mark.parametrize("name", MODULATIONS)
def test_modulations_bounded(name):
    value = modulation_fraction(name, **states())
    assert np.all((value >= 0) & (value <= 1))


def test_independent_axes_have_unique_candidate_ids():
    candidates = [TransportCandidate(m, law, ratio, alpha)
                  for m in MODULATIONS for law, alpha in [("Darcy", 0.), ("Forchheimer", 1.)]
                  for ratio in (1., 3.)]
    assert len({c.candidate_id for c in candidates}) == 20


def test_configuration_preserves_unrelated_transport_parameters():
    base = replace(NozzleTransport(), oxygen_diffusivity_closed=2.e-7)
    updated = TransportCandidate("crack_only", "Forchheimer", 3., 4.).configure_transport(base)
    assert updated.oxygen_diffusivity_closed == base.oxygen_diffusivity_closed
    assert updated.forchheimer_inverse_flux_scale == 4.
    assert updated.meridional_to_through_diffusivity_ratio == 3.


@pytest.mark.parametrize("law,alpha", [("Darcy", 0.), ("Forchheimer", 2000.)])
def test_opt_in_combined_candidate_preserves_entire_short_forward_solution(law, alpha):
    grid = NozzleGrid(meridional_cells=5, thickness_cells=3, time_step=.01, output_interval=.1)
    mission = NozzleMission(duration=.3)
    transport = replace(NozzleTransport(), forchheimer_inverse_flux_scale=alpha,
                        meridional_to_through_diffusivity_ratio=3.)
    old = simulate_reusable_nozzle_mission("old", grid=grid, mission=mission, transport=transport)
    new = simulate_reusable_nozzle_mission("new", grid=grid, mission=mission, transport=transport,
        transport_candidate=TransportCandidate("combined_opening", law, 3., alpha))
    for name in old.states:
        np.testing.assert_array_equal(old.states[name], new.states[name])


def test_alternative_modulation_is_used_in_actual_forward_solver():
    grid = NozzleGrid(meridional_cells=5, thickness_cells=3, time_step=.01, output_interval=.1)
    mission = NozzleMission(duration=.3)
    results = [simulate_reusable_nozzle_mission(m, grid=grid, mission=mission,
        transport_candidate=TransportCandidate(m, "Darcy")) for m in ("constant", "combined_opening")]
    assert np.max(np.abs(results[0].states["pH2O"] - results[1].states["pH2O"])) > 0


@pytest.mark.parametrize("kwargs", [
    dict(modulation="unknown", flux_law="Darcy"),
    dict(modulation="constant", flux_law="Darcy", inverse_flux_scale=1.),
    dict(modulation="constant", flux_law="Forchheimer", inverse_flux_scale=0.),
    dict(modulation="constant", flux_law="Darcy", directional_ratio=-1.),
    dict(modulation="constant", flux_law="Darcy", directional_ratio=float("nan")),
])
def test_invalid_physical_candidates_rejected(kwargs):
    with pytest.raises(ValueError):
        TransportCandidate(**kwargs)
