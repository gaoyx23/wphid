"""Pre-inference diagnostics for Arrhenius parameter identifiability."""

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from wphid.types import Trajectory


def diagnose_arrhenius_identifiability(
    trajectories: Sequence[Trajectory], theta_candidates: Sequence[float]
) -> dict[str, object]:
    """Quantify temperature excitation before fitting Arrhenius candidates.

    The result uses observed temperatures only. It reports standardized
    candidate-column conditioning and correlation, the material-wall
    temperature span, and the smallest separation between condition-level
    Arrhenius profile scores.
    """
    theta = np.asarray(tuple(theta_candidates), dtype=float)
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError("at least two Arrhenius candidates are required")
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0.0):
        raise ValueError("Arrhenius candidates must be finite and positive")
    if np.unique(theta).size != theta.size:
        raise ValueError("Arrhenius candidates must be unique")

    profiles = tuple(trajectories)
    if len(profiles) < 2:
        raise ValueError("at least two trajectories are required for profile diagnostics")

    temperatures = []
    profile_scores = []
    for trajectory in profiles:
        try:
            temperature = np.asarray(trajectory.states["T"], dtype=float)
        except KeyError as error:
            raise ValueError("every trajectory must expose a temperature state 'T'") from error
        if temperature.ndim != 1 or temperature.size < 2:
            raise ValueError("temperature trajectories must be one-dimensional")
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("temperature trajectories must be finite and positive")
        columns = np.exp(-np.outer(1.0 / temperature, theta))
        temperatures.append(temperature)
        profile_scores.append(columns.mean(axis=0))

    all_temperatures = np.concatenate(temperatures)
    candidate_columns = np.exp(-np.outer(1.0 / all_temperatures, theta))
    centered = candidate_columns - candidate_columns.mean(axis=0)
    scale = centered.std(axis=0)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("temperature excitation is insufficient to standardize candidates")
    standardized = centered / scale
    correlation = np.corrcoef(standardized, rowvar=False)

    scores = np.vstack(profile_scores)
    score_norm = np.linalg.norm(scores, axis=1)
    if np.any(score_norm <= np.finfo(float).eps):
        raise ValueError("Arrhenius profile scores must be non-zero")
    normalized_scores = scores / score_norm[:, np.newaxis]
    separation = min(
        float(np.linalg.norm(normalized_scores[left] - normalized_scores[right]))
        for left, right in combinations(range(len(profiles)), 2)
    )

    minimum = float(all_temperatures.min())
    maximum = float(all_temperatures.max())
    return {
        "condition_ids": tuple(trajectory.condition_id for trajectory in profiles),
        "theta_candidates": theta.copy(),
        "arrhenius_column_correlation": correlation,
        "standardized_condition_number": float(np.linalg.cond(standardized)),
        "temperature_span": {
            "minimum": minimum,
            "maximum": maximum,
            "span": maximum - minimum,
        },
        "profile_score_separation": separation,
    }
