"""Exact representation and failure cases, with artificial source arrays."""

import numpy as np
import pytest

from wphid.source_bundle_v1 import build_bundle_design, decode_source_coefficients
from wphid.source_geometry_v6 import profile_information


def fixture():
    rates = np.array([[1., 2., .5], [.3, 1.5, 3.]])
    fraction = np.array([0., .4, 1.])
    weights = {name: np.array([[.2, .5, .3], [1., -.4, .2]])
               for name in ("carbon", "pO2", "pH2O", "pProducts", "T")}
    return weights, rates, fraction


def test_unknown_conversions_are_exactly_representable_and_decodable():
    weights, rates, fraction = fixture()
    ao, aw = .7, 1.4
    co, cw, cp, heat = 3., 5., 4., 11.
    design = build_bundle_design(weights, rates, relative_rate=aw/ao,
                                 heat_fraction=fraction, heat_contrast=.8)
    coefficients = np.array([ao, co*ao, cw*aw, cp*ao, heat*ao])
    oxygen_rate, steam_rate = ao*rates[0], aw*rates[1]
    direct_sources = {
        "carbon": -(oxygen_rate+steam_rate), "pO2": -co*oxygen_rate,
        "pH2O": -cw*steam_rate, "pProducts": cp*(oxygen_rate+steam_rate),
        "T": heat*(oxygen_rate+steam_rate)/(1+.8*fraction),
    }
    expected = np.concatenate([weights[field] @ direct_sources[field] for field in weights])
    np.testing.assert_allclose(design.matrix @ coefficients, expected, rtol=1e-12)
    decoded = decode_source_coefficients(design, coefficients, relative_rate=aw/ao)
    assert decoded == pytest.approx({
        "oxygen_rate_scale": ao, "steam_rate_scale": aw,
        "oxygen_conversion": co, "steam_conversion": cw, "product_conversion": cp,
        "heat_release_over_density_ambient_capacity": heat,
    })


@pytest.mark.parametrize("coordinate", ["relative_rate", "heat_contrast"])
def test_outer_derivative_includes_temperature_loading_and_rate_mixture(coordinate):
    weights, rates, fraction = fixture()
    parameters = {"relative_rate": 1.3, "heat_contrast": .6}
    central = build_bundle_design(weights, rates, heat_fraction=fraction, **parameters)
    upper, lower = dict(parameters), dict(parameters)
    step = 1e-6
    upper[coordinate] += step
    lower[coordinate] -= step
    finite_difference = (
        build_bundle_design(weights, rates, heat_fraction=fraction, **upper).matrix
        - build_bundle_design(weights, rates, heat_fraction=fraction, **lower).matrix
    )/(2*step)
    analytic = (central.derivative_relative_rate if coordinate == "relative_rate"
                else central.derivative_heat_contrast)
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-9)


def test_missing_carbon_anchor_retains_relative_source_model_but_blocks_absolute_scales():
    weights, rates, fraction = fixture()
    del weights["carbon"]
    design = build_bundle_design(weights, rates, relative_rate=1.3,
                                 heat_fraction=fraction, heat_contrast=.6)
    assert not design.carbon_scale_anchor_present_in_model
    with pytest.raises(ValueError, match="anchor"):
        decode_source_coefficients(design, np.ones(4), relative_rate=1.3)


def test_identical_reaction_shapes_still_allow_complete_ratio_compensation():
    weights, _, fraction = fixture()
    rates = np.array([[1., 2., 3.], [1., 2., 3.]])
    design = build_bundle_design(weights, rates, relative_rate=1.3,
                                 heat_fraction=fraction, heat_contrast=.6)
    derivative = design.derivative_relative_rate @ np.ones(5)
    diagnostic = profile_information(derivative[:, None], design.matrix)
    assert diagnostic.rank == 0


def test_crack_response_cannot_be_silently_bound_to_reaction_sources():
    weights, rates, fraction = fixture()
    weights["crack"] = np.ones((1, 3))
    with pytest.raises(ValueError, match="unqualified"):
        build_bundle_design(weights, rates, relative_rate=1.3,
                            heat_fraction=fraction, heat_contrast=.6)


def test_null_or_zero_amplitude_is_not_decoded_as_an_active_precise_reaction():
    weights, rates, fraction = fixture()
    design = build_bundle_design(weights, rates, relative_rate=1.3,
                                 heat_fraction=fraction, heat_contrast=.6)
    with pytest.raises(ValueError, match="positive"):
        decode_source_coefficients(design, np.array([0., 1., 1., 1., 1.]), relative_rate=1.3)
