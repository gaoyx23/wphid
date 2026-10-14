import numpy as np
import pytest

from wphid.eiv import fit_simex_fixed_support


def test_fixed_support_simex_is_deterministic_for_callback_reconstructions():
    x = np.linspace(-2.0, 2.0, 60)
    target = 0.4 + 1.75 * x
    levels = np.array([0.0, 0.5, 1.0])

    def rebuild(level: float, replicate: int):
        nuisance = np.cos(x + level + replicate)
        return np.column_stack((nuisance, x)), target

    first = fit_simex_fixed_support(levels, rebuild, support=np.array([1]), n_replicates=3)
    second = fit_simex_fixed_support(levels, rebuild, support=np.array([1]), n_replicates=3)

    np.testing.assert_array_equal(first.support, np.array([1]))
    np.testing.assert_allclose(first.coefficient_path, second.coefficient_path)
    np.testing.assert_allclose(
        first.extrapolated_coefficients, second.extrapolated_coefficients
    )
    assert first.diagnostics["extrapolation"]["stable"] is True


def test_no_noise_reconstructions_match_fixed_support_ols():
    rng = np.random.default_rng(209)
    matrix = rng.normal(size=(400, 3))
    support = np.array([0, 2])
    target = matrix[:, support] @ np.array([1.25, -0.4])
    expected = np.linalg.lstsq(matrix[:, support], target, rcond=None)[0]
    levels = np.array([0.0, 0.5, 1.0])
    reconstructions = tuple(
        tuple((matrix, target) for _ in range(2)) for _ in levels
    )

    result = fit_simex_fixed_support(levels, reconstructions, support)

    np.testing.assert_allclose(
        result.coefficient_path, np.tile(expected, (levels.size, 1)), atol=1e-12
    )
    np.testing.assert_allclose(result.extrapolated_coefficients, expected, atol=1e-12)
    np.testing.assert_allclose(result.coefficient_standard_errors, 0.0, atol=1e-15)


def test_simex_extrapolation_reduces_known_attenuation_bias():
    rng = np.random.default_rng(31)
    n_observations = 12_000
    coefficient = 2.5
    latent = rng.normal(size=n_observations)
    clean_target = coefficient * latent
    noise_scale = 0.75
    target_noise_scale = 0.1
    levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    reconstructions = []
    for level in levels:
        replicates = []
        for _ in range(6):
            contaminated = latent + np.sqrt(1.0 + level) * noise_scale * rng.normal(
                size=n_observations
            )
            noisy_target = clean_target + np.sqrt(1.0 + level) * target_noise_scale * rng.normal(
                size=n_observations
            )
            # The zero column demonstrates that refitting never selects support.
            replicates.append(
                (np.column_stack((np.zeros(n_observations), contaminated)), noisy_target)
            )
        reconstructions.append(tuple(replicates))

    result = fit_simex_fixed_support(levels, tuple(reconstructions), support=np.array([1]))
    uncorrected_error = abs(result.coefficient_path[0, 0] - coefficient)
    corrected_error = abs(result.extrapolated_coefficients[0] - coefficient)

    assert result.diagnostics["extrapolation"]["stable"] is True
    assert corrected_error < uncorrected_error


def test_ill_conditioned_extrapolation_falls_back_to_uncorrected_level_zero():
    x = np.linspace(-1.0, 1.0, 100)
    target = 3.0 * x
    levels = np.array([0.0, 0.5, 1.0])
    reconstructions = tuple(((x[:, None], target),) for _ in levels)

    result = fit_simex_fixed_support(
        levels,
        reconstructions,
        support=np.array([0]),
        extrapolation_condition_limit=1.0,
    )

    diagnostic = result.diagnostics["extrapolation"]
    assert diagnostic["stable"] is False
    assert diagnostic["fallback_reason"] == "ill_conditioned_extrapolation"
    assert diagnostic["fallback_level"] == pytest.approx(0.0)
    np.testing.assert_allclose(result.extrapolated_coefficients, result.coefficient_path[0])

@pytest.mark.parametrize(
    "selected",
    (
        np.column_stack((np.zeros(80), np.linspace(-1.0, 1.0, 80))),
        np.column_stack((np.linspace(-1.0, 1.0, 80), 2.0 * np.linspace(-1.0, 1.0, 80))),
    ),
    ids=("zero-column", "perfect-collinearity"),
)
def test_rank_deficient_fixed_support_is_non_identifiable(selected):
    levels = np.array([0.0, 0.5, 1.0])
    target = selected @ np.array([1.0, 2.0])
    reconstructions = tuple(((selected, target),) for _ in levels)

    result = fit_simex_fixed_support(levels, reconstructions, support=np.array([0, 1]))

    identifiability = result.diagnostics["identifiability"]
    extrapolation = result.diagnostics["extrapolation"]
    assert identifiability["identifiable"] is False
    assert extrapolation["stable"] is False
    assert extrapolation["fallback_reason"] == "non_identifiable_fixed_support"
    assert extrapolation["fallback_level"] == pytest.approx(0.0)
    np.testing.assert_allclose(result.extrapolated_coefficients, result.coefficient_path[0])


def test_levels_must_include_exact_zero():
    x = np.linspace(-1.0, 1.0, 30)
    reconstructions = tuple(((x[:, None], 2.0 * x),) for _ in range(3))

    with pytest.raises(ValueError, match="include 0.0"):
        fit_simex_fixed_support(
            np.array([0.1, 0.5, 1.0]), reconstructions, support=np.array([0])
        )


@pytest.mark.parametrize(
    "levels",
    (
        np.array([-0.5, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.5]),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.0, np.inf, 1.0]),
        np.array([[0.0, 0.5, 1.0]]),
    ),
)
def test_invalid_levels_are_rejected(levels):
    x = np.linspace(-1.0, 1.0, 30)
    reconstructions = tuple(((x[:, None], x),) for _ in range(3))

    with pytest.raises(ValueError):
        fit_simex_fixed_support(levels, reconstructions, support=np.array([0]))


@pytest.mark.parametrize(
    "support",
    (
        np.array([], dtype=int),
        np.array([-1]),
        np.array([True]),
        np.array([0.0]),
        np.array([1, 0]),
        np.array([0, 0]),
        np.array([2]),
    ),
)
def test_invalid_frozen_support_is_rejected(support):
    x = np.linspace(-1.0, 1.0, 30)
    matrix = np.column_stack((x, x**2))
    reconstructions = tuple(((matrix, x),) for _ in range(3))

    with pytest.raises(ValueError):
        fit_simex_fixed_support(
            np.array([0.0, 0.5, 1.0]), reconstructions, support=support
        )


def test_reconstructed_designs_must_keep_the_same_full_column_layout():
    x = np.linspace(-1.0, 1.0, 30)
    reconstructions = (
        ((np.column_stack((x, x**2)), x),),
        ((np.column_stack((x, x**2, x**3)), x),),
        ((np.column_stack((x, x**2)), x),),
    )

    with pytest.raises(ValueError, match="same full column layout"):
        fit_simex_fixed_support(
            np.array([0.0, 0.5, 1.0]), reconstructions, support=np.array([0])
        )

def test_one_rank_deficient_replicate_makes_fixed_support_non_identifiable():
    x = np.linspace(-1.0, 1.0, 80)
    full_rank = np.column_stack((x, x**2))
    rank_deficient = np.column_stack((x, 2.0 * x))
    target = full_rank @ np.array([1.0, -0.5])
    levels = np.array([0.0, 0.5, 1.0])
    reconstructions = (
        ((full_rank, target), (full_rank, target)),
        ((full_rank, target), (rank_deficient, target)),
        ((full_rank, target), (full_rank, target)),
    )

    result = fit_simex_fixed_support(levels, reconstructions, support=np.array([0, 1]))

    assert result.diagnostics["identifiability"]["identifiable"] is False
    np.testing.assert_array_equal(
        result.diagnostics["identifiability"]["rank_deficient_replicates"][1],
        np.array([1]),
    )
    assert result.diagnostics["extrapolation"]["stable"] is False
    np.testing.assert_allclose(result.extrapolated_coefficients, result.coefficient_path[0])
