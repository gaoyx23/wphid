import numpy as np

from experiments.paper_r16_crack_controlled_transport_recovery import _positive_fit
from experiments.paper_r17_shared_multispecies_transport_recovery import (
    _positive_orthogonal_fit,
    _rank_models,
    SPECIES,
)


def test_nonnegative_orthogonal_fit_recovers_clean_positive_coefficients():
    rng = np.random.default_rng(8101)
    matrix = rng.normal(size=(2000, 3))
    truth = np.array([0.4, 1.2, 2.5])
    target = matrix @ truth
    fitted = _positive_orthogonal_fit(matrix, target)
    np.testing.assert_allclose(fitted, truth, rtol=1e-6, atol=1e-8)


def test_nonnegative_orthogonal_fit_reduces_classical_attenuation():
    rng = np.random.default_rng(8102)
    latent = rng.normal(size=10000)
    truth = 2.0
    matrix = (latent + 0.55 * rng.normal(size=latent.size))[:, None]
    target = truth * latent + 0.08 * rng.normal(size=latent.size)
    ordinary = _positive_fit(matrix, target)[0]
    orthogonal = _positive_orthogonal_fit(matrix, target)[0]
    assert abs(orthogonal - truth) < abs(ordinary - truth)


def test_reporting_refit_failure_does_not_delete_cross_validated_family():
    per_condition = {}
    for index, name in enumerate(("a", "b", "c"), start=1):
        per_condition[name] = {
            "family": {
                species: (
                    np.array([[float(index)]]),
                    np.array([2.0 * index]),
                )
                for species in SPECIES
            }
        }

    def diagnostic_fit(matrix, target):
        if matrix.shape[0] == 3:
            return None, {"success": False, "reason": "report_only_failure"}
        return np.array([2.0]), {"success": True}

    ranking, _, _, _, diagnostics = _rank_models(
        per_condition,
        fit_function=lambda matrix, target: np.array([2.0]),
        diagnostic_fit_function=diagnostic_fit,
        return_fit_diagnostics=True,
    )

    assert len(ranking) == 1
    assert ranking[0]["family"] == "family"
    assert ranking[0]["coefficient_refit_complete"] is False
    assert all(value is None for value in ranking[0]["coefficients_by_species"].values())
    assert diagnostics["failure_count"] == len(SPECIES)
