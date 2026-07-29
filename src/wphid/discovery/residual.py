"""Cross-condition weak-residual evidence for controlled library expansion."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from wphid.pipeline import _condition_matrix, _normalized_rmse, discover_closed
from wphid.types import Candidate, DiscoveryResult, Trajectory


_MIN_ABSOLUTE_CORRELATION = 0.25
_MIN_SIGN_CONSISTENCY = 0.75
_MIN_RELATIVE_VALIDATION_REDUCTION = 0.01


def _readonly_vector(values: np.ndarray, label: str) -> np.ndarray:
    vector = np.array(values, dtype=float, copy=True)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError(
            f"{label} must be a one-dimensional vector with at least two rows"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class ResidualBlock:
    """One condition's weak residual and aligned expansion columns."""

    residual: np.ndarray
    weak_columns: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        residual = _readonly_vector(self.residual, "residual")
        columns = {}
        for name, values in self.weak_columns.items():
            column = _readonly_vector(values, f"weak column {name!r}")
            if column.shape != residual.shape:
                raise ValueError(
                    f"weak column {name!r} must have the same rows as residual"
                )
            columns[name] = column
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "weak_columns", MappingProxyType(columns))


@dataclass(frozen=True)
class ExpansionScore:
    """Structured residual evidence for one admissible grammar candidate."""

    candidate: Candidate
    correlations_by_condition: Mapping[str, float]
    median_absolute_correlation: float
    sign_consistency: float
    stable: bool
    residual_score: float
    validation_error_reduction: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correlations_by_condition",
            MappingProxyType(dict(self.correlations_by_condition)),
        )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def rank_expansions(
    residuals_by_condition: Mapping[str, ResidualBlock],
    base_candidates: Sequence[Candidate],
    expansion_candidates: Sequence[Candidate],
) -> list[ExpansionScore]:
    """Rank novel candidates using stable weak-residual evidence only.

    Ranking consumes training-condition residuals. Held-out validation error
    is deliberately absent from this API and is used later only to decide
    whether a highly ranked candidate is admitted.
    """
    if len(residuals_by_condition) < 2:
        raise ValueError("cross-condition ranking requires at least two conditions")
    base_names = {candidate.name for candidate in base_candidates}
    if len(base_names) != len(base_candidates):
        raise ValueError("base candidate names must be unique")
    expansion_names = [candidate.name for candidate in expansion_candidates]
    if len(set(expansion_names)) != len(expansion_names):
        raise ValueError("expansion candidate names must be unique")

    required_conditions = len(residuals_by_condition) // 2 + 1
    scores = []
    for candidate in expansion_candidates:
        if candidate.name in base_names:
            continue
        correlations = {}
        for condition_id, block in residuals_by_condition.items():
            try:
                column = block.weak_columns[candidate.name]
            except KeyError as error:
                raise ValueError(
                    f"condition {condition_id!r} lacks weak column "
                    f"{candidate.name!r}"
                ) from error
            correlations[condition_id] = _correlation(block.residual, column)

        values = np.asarray(tuple(correlations.values()))
        absolute_values = np.abs(values)
        evidence = absolute_values >= _MIN_ABSOLUTE_CORRELATION
        evidence_signs = np.sign(values[evidence])
        if evidence_signs.size:
            positive = int(np.count_nonzero(evidence_signs > 0.0))
            negative = int(np.count_nonzero(evidence_signs < 0.0))
            sign_consistency = max(positive, negative) / evidence_signs.size
        else:
            sign_consistency = 0.0
        median_absolute_correlation = float(np.median(absolute_values))
        stable = bool(
            np.count_nonzero(evidence) >= required_conditions
            and sign_consistency >= _MIN_SIGN_CONSISTENCY
        )
        residual_score = median_absolute_correlation * sign_consistency
        scores.append(
            ExpansionScore(
                candidate=candidate,
                correlations_by_condition=correlations,
                median_absolute_correlation=median_absolute_correlation,
                sign_consistency=sign_consistency,
                stable=stable,
                residual_score=residual_score,
            )
        )

    return sorted(
        scores,
        key=lambda score: (
            not score.stable,
            -score.residual_score,
            score.candidate.name,
        ),
    )


def _complexity(candidate: Candidate) -> int:
    if candidate.group.startswith("G") and candidate.group[1:].isdigit():
        return int(candidate.group[1:])
    return max(1, len(candidate.parents))


def _training_error(
    trajectories_by_id: Mapping[str, Trajectory],
    condition_ids: Sequence[str],
    candidates: Sequence[Candidate],
    coefficients: np.ndarray,
) -> float:
    matrices = [
        _condition_matrix(trajectories_by_id[condition_id], candidates)
        for condition_id in condition_ids
    ]
    matrix = np.vstack([item[0] for item in matrices])
    target = np.concatenate([item[1] for item in matrices])
    return _normalized_rmse(matrix, target, coefficients)


def _residual_blocks(
    trajectories_by_id: Mapping[str, Trajectory],
    condition_ids: Sequence[str],
    current_candidates: Sequence[Candidate],
    expansion_candidates: Sequence[Candidate],
    coefficients: np.ndarray,
) -> dict[str, ResidualBlock]:
    blocks = {}
    for condition_id in condition_ids:
        trajectory = trajectories_by_id[condition_id]
        matrix, target = _condition_matrix(trajectory, current_candidates)
        weak_columns = {
            candidate.name: _condition_matrix(trajectory, [candidate])[0][:, 0]
            for candidate in expansion_candidates
        }
        blocks[condition_id] = ResidualBlock(
            residual=target - matrix @ coefficients,
            weak_columns=weak_columns,
        )
    return blocks


def _score_diagnostic(score: ExpansionScore) -> dict[str, Any]:
    return {
        "candidate": score.candidate.name,
        "complexity_layer": _complexity(score.candidate),
        "correlations_by_condition": dict(score.correlations_by_condition),
        "median_absolute_correlation": score.median_absolute_correlation,
        "sign_consistency": score.sign_consistency,
        "stable": score.stable,
        "residual_score": score.residual_score,
        "validation_error_reduction": score.validation_error_reduction,
    }


def discover_with_expansion(
    trajectories: Iterable[Trajectory],
    base_candidates: Sequence[Candidate],
    expansion_candidates: Sequence[Candidate],
    train_ids: Iterable[str],
    validation_ids: Iterable[str],
    thresholds: Iterable[float],
    max_rounds: int = 2,
) -> DiscoveryResult:
    """Fit a base library and admit only stable, held-out-improving expansions.

    Weak-residual evidence is computed independently in every training
    condition. Candidates are considered one grammar-complexity layer per
    round; held-out conditions are consulted only after ranking, and a
    candidate that fails to reduce validation error is rejected.
    """
    if not isinstance(max_rounds, (int, np.integer)) or max_rounds < 0:
        raise ValueError("max_rounds must be a non-negative integer")

    observations = tuple(trajectories)
    current_candidates = list(base_candidates)
    available_candidates = list(expansion_candidates)
    training_ids = tuple(train_ids)
    held_out_ids = tuple(validation_ids)
    threshold_values = tuple(thresholds)
    if len({candidate.name for candidate in available_candidates}) != len(
        available_candidates
    ):
        raise ValueError("expansion candidate names must be unique")
    current_names = {candidate.name for candidate in current_candidates}
    available_candidates = [
        candidate
        for candidate in available_candidates
        if candidate.name not in current_names
    ]

    current_result = discover_closed(
        trajectories=observations,
        candidates=current_candidates,
        train_ids=training_ids,
        validation_ids=held_out_ids,
        thresholds=threshold_values,
    )
    fixed_validation_error = current_result.validation_error
    trajectories_by_id = {
        trajectory.condition_id: trajectory for trajectory in observations
    }
    fixed_training_error = _training_error(
        trajectories_by_id,
        training_ids,
        current_candidates,
        current_result.coefficients,
    )

    rounds: list[dict[str, Any]] = []
    admitted_candidates: list[str] = []
    complexity_layers = sorted(
        {_complexity(candidate) for candidate in available_candidates}
    )
    for round_index, layer in enumerate(complexity_layers[:max_rounds], start=1):
        layer_candidates = [
            candidate
            for candidate in available_candidates
            if _complexity(candidate) == layer
        ]
        residuals = _residual_blocks(
            trajectories_by_id,
            training_ids,
            current_candidates,
            layer_candidates,
            current_result.coefficients,
        )
        ranked = rank_expansions(
            residuals,
            base_candidates=current_candidates,
            expansion_candidates=layer_candidates,
        )
        updated_scores: dict[str, ExpansionScore] = {
            score.candidate.name: score for score in ranked
        }
        trial_results: dict[str, tuple[DiscoveryResult, float, float]] = {}
        current_training_error = _training_error(
            trajectories_by_id,
            training_ids,
            current_candidates,
            current_result.coefficients,
        )
        for score in ranked:
            if not score.stable:
                continue
            trial_candidates = [*current_candidates, score.candidate]
            trial_result = discover_closed(
                trajectories=observations,
                candidates=trial_candidates,
                train_ids=training_ids,
                validation_ids=held_out_ids,
                thresholds=threshold_values,
            )
            validation_reduction = (
                current_result.validation_error - trial_result.validation_error
            )
            trial_training_error = _training_error(
                trajectories_by_id,
                training_ids,
                trial_candidates,
                trial_result.coefficients,
            )
            training_reduction = current_training_error - trial_training_error
            updated_scores[score.candidate.name] = replace(
                score,
                validation_error_reduction=validation_reduction,
            )
            trial_results[score.candidate.name] = (
                trial_result,
                validation_reduction,
                training_reduction,
            )

        minimum_reduction = (
            _MIN_RELATIVE_VALIDATION_REDUCTION * current_result.validation_error
        )
        admissible = []
        for score in ranked:
            trial = trial_results.get(score.candidate.name)
            if trial is None:
                continue
            trial_result, validation_reduction, training_reduction = trial
            selected_names = {
                trial_result.names[index] for index in trial_result.support
            }
            if (
                score.candidate.name in selected_names
                and validation_reduction > minimum_reduction
            ):
                admissible.append(
                    (
                        validation_reduction,
                        score.residual_score,
                        -trial_result.validation_error,
                        score.candidate.name,
                        score.candidate,
                        trial_result,
                        training_reduction,
                    )
                )

        admitted_name = None
        admitted_training_reduction = None
        admitted_validation_reduction = None
        if admissible:
            (
                admitted_validation_reduction,
                _,
                _,
                admitted_name,
                admitted_candidate,
                current_result,
                admitted_training_reduction,
            ) = max(admissible, key=lambda item: item[:4])
            current_candidates.append(admitted_candidate)
            admitted_candidates.append(admitted_name)
            available_candidates = [
                candidate
                for candidate in available_candidates
                if candidate.name != admitted_name
            ]

        rounds.append(
            {
                "round": round_index,
                "complexity_layer": layer,
                "residual_condition_ids": training_ids,
                "residual_ranking": [
                    _score_diagnostic(updated_scores[score.candidate.name])
                    for score in ranked
                ],
                "admitted_candidate": admitted_name,
                "admitted_training_error_reduction": admitted_training_reduction,
                "admitted_validation_error_reduction": (
                    admitted_validation_reduction
                ),
                "minimum_validation_error_reduction": minimum_reduction,
            }
        )

    final_training_error = _training_error(
        trajectories_by_id,
        training_ids,
        current_candidates,
        current_result.coefficients,
    )
    diagnostics = {
        **dict(current_result.diagnostics),
        "fixed_library_validation_error": fixed_validation_error,
        "fixed_library_training_error": fixed_training_error,
        "final_training_error": final_training_error,
        "initial_candidate_names": tuple(
            candidate.name for candidate in base_candidates
        ),
        "expansion_candidate_names": tuple(
            candidate.name for candidate in expansion_candidates
        ),
        "admitted_expansions": tuple(admitted_candidates),
        "expansion_rounds": rounds,
    }
    return DiscoveryResult(
        names=current_result.names,
        coefficients=current_result.coefficients,
        support=current_result.support,
        validation_error=current_result.validation_error,
        diagnostics=diagnostics,
    )
