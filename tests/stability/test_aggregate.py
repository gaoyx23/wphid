import numpy as np
import pytest

from wphid.stability.aggregate import StabilitySummary, aggregate_stability


def test_aggregate_reports_inclusion_sign_consistency_and_stable_support():
    summary = aggregate_stability(
        coefficients=np.array(
            [
                [1.0, 0.0, -2.0],
                [0.8, 0.0, 1.5],
                [1.1, 0.0, -1.0],
                [0.9, 0.0, 0.0],
            ]
        ),
        candidate_names=("stable", "absent", "mixed"),
        inclusion_threshold=0.75,
        sign_consistency_threshold=0.75,
    )

    np.testing.assert_allclose(summary.inclusion_probabilities, [1.0, 0.0, 0.75])
    np.testing.assert_allclose(summary.sign_consistency, [1.0, 0.0, 2.0 / 3.0])
    np.testing.assert_array_equal(summary.support, np.array([0]))
    assert summary.names == ("stable", "absent", "mixed")


def test_aggregate_reports_train_only_parameter_selection_frequencies():
    summary = aggregate_stability(
        coefficients=np.array([[1.0], [1.0], [0.0], [1.0]]),
        candidate_names=("factor",),
        theta_values=(1200.0, 1400.0, None, 1200.0),
    )

    assert dict(summary.parameter_selection_frequencies) == {
        1200.0: 2.0 / 3.0,
        1400.0: 1.0 / 3.0,
    }

@pytest.mark.parametrize(
    ("inclusion", "signs"),
    [
        (np.array([np.nan]), np.array([1.0])),
        (np.array([1.0]), np.array([np.inf])),
    ],
)


def test_stability_summary_rejects_nonfinite_probability_vectors(
    inclusion, signs
):
    with pytest.raises(ValueError, match="finite probabilities"):
        StabilitySummary(
            names=("factor",),
            inclusion_probabilities=inclusion,
            sign_consistency=signs,
            support=np.array([0]),
            parameter_selection_frequencies={},
            n_resamples=1,
            inclusion_threshold=0.8,
            sign_consistency_threshold=0.8,
            selection_tolerance=0.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inclusion_threshold", np.nan),
        ("sign_consistency_threshold", np.inf),
        ("selection_tolerance", np.nan),
    ],
)


def test_stability_summary_rejects_nonfinite_thresholds(field, value):
    arguments = {
        "names": ("factor",),
        "inclusion_probabilities": np.array([1.0]),
        "sign_consistency": np.array([1.0]),
        "support": np.array([0]),
        "parameter_selection_frequencies": {},
        "n_resamples": 1,
        "inclusion_threshold": 0.8,
        "sign_consistency_threshold": 0.8,
        "selection_tolerance": 0.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="finite"):
        StabilitySummary(**arguments)


def test_stability_summary_rejects_nonfinite_parameter_frequency_keys():
    with pytest.raises(ValueError, match="parameter selections"):
        StabilitySummary(
            names=("factor",),
            inclusion_probabilities=np.array([1.0]),
            sign_consistency=np.array([1.0]),
            support=np.array([0]),
            parameter_selection_frequencies={np.nan: 1.0},
            n_resamples=1,
            inclusion_threshold=0.8,
            sign_consistency_threshold=0.8,
            selection_tolerance=0.0,
        )