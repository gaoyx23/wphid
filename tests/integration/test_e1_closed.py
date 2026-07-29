import json

import numpy as np

from wphid.pipeline import discover_closed
from wphid.truth.aging_ode import simulate_named_condition
from wphid.types import Candidate


EXPECTED_SUPPORT = {
    "arr(T)*pO2",
    "arr(T)*pO2*sigma",
    "arr(T)*pO2*Df",
}


def _closed_candidate_library():
    def arrhenius(states):
        return np.exp(-1800.0 / states["T"])

    return [
        Candidate(
            "arr(T)*pO2",
            lambda states: arrhenius(states) * states["pO2"],
            (),
            "G2",
        ),
        Candidate(
            "arr(T)*pO2*sigma",
            lambda states: arrhenius(states) * states["pO2"] * states["sigma"],
            (),
            "G3",
        ),
        Candidate(
            "arr(T)*pO2*Df",
            lambda states: arrhenius(states) * states["pO2"] * states["Df"],
            (),
            "G3",
        ),
        Candidate("pO2", lambda states: states["pO2"], (), "F"),
        Candidate(
            "pO2*sigma",
            lambda states: states["pO2"] * states["sigma"],
            (),
            "G2",
        ),
        Candidate(
            "arr(T)*pO2*sat(Df)",
            lambda states: arrhenius(states)
            * states["pO2"]
            / (1.0 - states["Df"]),
            (),
            "G3",
        ),
    ]


def test_closed_pipeline_recovers_three_oxidation_terms_without_spurious_support():
    """Skipping validation selection, weak fitting, or sparsity breaks E1 recovery."""
    condition_ids = ("thermal_hold", "fatigue_rich", "oxygen_rich")
    trajectories = [simulate_named_condition(name) for name in condition_ids]

    result = discover_closed(
        trajectories=trajectories,
        candidates=_closed_candidate_library(),
        train_ids=("thermal_hold", "fatigue_rich"),
        validation_ids=("oxygen_rich",),
        thresholds=np.geomspace(1e-4, 0.5, 48),
    )

    selected_names = {result.names[index] for index in result.support}
    assert selected_names == EXPECTED_SUPPORT
    assert result.validation_error < 0.03


def test_e1_experiment_writes_fixed_configuration_metrics(tmp_path):
    """Dropping reproducible configuration or metrics persistence breaks E1."""
    from experiments.e1_closed_library import run_experiment

    metrics = run_experiment(tmp_path)
    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert persisted == metrics
    assert metrics["seed"] == 20260729
    assert metrics["configuration"]["train_ids"] == [
        "thermal_hold",
        "fatigue_rich",
    ]
    assert metrics["configuration"]["validation_ids"] == ["oxygen_rich"]
    assert set(metrics["support"]) == EXPECTED_SUPPORT
    assert metrics["validation_normalized_rmse"] < 0.03
