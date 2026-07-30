"""Cross-condition weak-residual evidence for controlled library expansion."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from wphid.pipeline import _condition_matrix, _normalized_rmse, discover_closed
from wphid.sparse.stlsq import fit_stlsq
from wphid.types import Candidate, DiscoveryResult, Trajectory


_MIN_ABSOLUTE_CORRELATION = 0.25
_MIN_SIGN_CONSISTENCY = 0.75
_MIN_RELATIVE_VALIDATION_REDUCTION = 0.01
_RESIDUAL_FIT_SELECTION_RULE = (
    "sparsest_nonempty_training_support_then_training_error"
)


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


@dataclass(frozen=True)
class _TrainingFit:
    """A conservative sparse fit selected without held-out conditions."""

    coefficients: np.ndarray
    support: np.ndarray
    error: float
    selected_threshold: float


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


def _training_matrix(
    trajectories_by_id: Mapping[str, Trajectory],
    condition_ids: Sequence[str],
    candidates: Sequence[Candidate],
) -> tuple[np.ndarray, np.ndarray]:
    matrices = [
        _condition_matrix(trajectories_by_id[condition_id], candidates)
        for condition_id in condition_ids
    ]
    return (
        np.vstack([item[0] for item in matrices]),
        np.concatenate([item[1] for item in matrices]),
    )


def _training_error(
    trajectories_by_id: Mapping[str, Trajectory],
    condition_ids: Sequence[str],
    candidates: Sequence[Candidate],
    coefficients: np.ndarray,
) -> float:
    matrix, target = _training_matrix(
        trajectories_by_id, condition_ids, candidates
    )
    return _normalized_rmse(matrix, target, coefficients)


def _fit_training_only(
    trajectories_by_id: Mapping[str, Trajectory],
    condition_ids: Sequence[str],
    candidates: Sequence[Candidate],
    thresholds: Sequence[float],
) -> _TrainingFit:
    """Choose a conservative residual base using training rows only.

    Missing-structure diagnostics require the base model not to absorb omitted
    interactions through flexible distractors. The selection therefore
    prefers the sparsest non-empty training support, then lower training
    error, then the higher threshold. If every fit is empty, the lowest-error
    empty fit is retained.
    """
    matrix, target = _training_matrix(
        trajectories_by_id, condition_ids, candidates
    )
    selected_key: tuple[bool, int, float, float] | None = None
    selected_fit: DiscoveryResult | None = None
    selected_threshold = float("nan")
    selected_error = float("nan")
    for threshold in thresholds:
        fit = fit_stlsq(matrix, target, float(threshold))
        training_error = _normalized_rmse(
            matrix, target, fit.coefficients
        )
        key = (
            fit.support.size == 0,
            int(fit.support.size),
            training_error,
            -float(threshold),
        )
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_fit = fit
            selected_threshold = float(threshold)
            selected_error = training_error

    if selected_fit is None:
        raise ValueError("thresholds must be non-empty")
    return _TrainingFit(
        coefficients=np.asarray(selected_fit.coefficients),
        support=np.asarray(selected_fit.support),
        error=selected_error,
        selected_threshold=selected_threshold,
    )


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


def _validate_discovery_inputs(
    observations: Sequence[Trajectory],
    current_candidates: Sequence[Candidate],
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
    test_ids: Sequence[str],
    threshold_values: np.ndarray,
) -> dict[str, Trajectory]:
    if not observations:
        raise ValueError("at least one trajectory is required")
    if not current_candidates:
        raise ValueError("at least one candidate is required")
    if len(training_ids) < 2:
        raise ValueError(
            "cross-condition expansion requires at least two training IDs"
        )
    roles = {
        "train_ids": training_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
    }
    for role, condition_ids in roles.items():
        if not condition_ids:
            raise ValueError(f"{role} must be non-empty")
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError(f"{role} must contain unique condition IDs")
    role_sets = [set(condition_ids) for condition_ids in roles.values()]
    if any(
        left & right
        for index, left in enumerate(role_sets)
        for right in role_sets[index + 1 :]
    ):
        raise ValueError(
            "train_ids, validation_ids, and test_ids must be pairwise disjoint"
        )
    trajectories_by_id = {
        trajectory.condition_id: trajectory for trajectory in observations
    }
    if len(trajectories_by_id) != len(observations):
        raise ValueError("trajectory condition IDs must be unique")
    requested_ids = set().union(*role_sets)
    missing = requested_ids.difference(trajectories_by_id)
    if missing:
        raise ValueError(f"unknown condition IDs: {', '.join(sorted(missing))}")
    names = [candidate.name for candidate in current_candidates]
    if len(set(names)) != len(names):
        raise ValueError("base candidate names must be unique")
    if threshold_values.ndim != 1 or threshold_values.size == 0:
        raise ValueError(
            "thresholds must be a non-empty one-dimensional sequence"
        )
    if not np.all(np.isfinite(threshold_values)) or np.any(
        threshold_values < 0.0
    ):
        raise ValueError(
            "thresholds must contain finite non-negative values"
        )
    return trajectories_by_id


def discover_with_expansion(
    trajectories: Iterable[Trajectory],
    base_candidates: Sequence[Candidate],
    expansion_candidates: Sequence[Candidate],
    train_ids: Iterable[str],
    validation_ids: Iterable[str],
    test_ids: Iterable[str],
    target_dimension: tuple[int, ...],
    thresholds: Iterable[float],
    max_rounds: int = 2,
) -> DiscoveryResult:
    """Screen on training, admit from a finite pool, then test once.

    Every residual and stability score uses training conditions only.
    Validation targets are consulted only for finite-pool admission and sparse
    threshold selection. Test targets are first read after all model choices
    and coefficients are frozen.
    """
    if not isinstance(max_rounds, (int, np.integer)) or max_rounds < 0:
        raise ValueError("max_rounds must be a non-negative integer")

    observations = tuple(trajectories)
    current_candidates = list(base_candidates)
    available_candidates = list(expansion_candidates)
    training_ids = tuple(train_ids)
    held_out_ids = tuple(validation_ids)
    independent_test_ids = tuple(test_ids)
    threshold_array = np.asarray(tuple(thresholds), dtype=float)
    trajectories_by_id = _validate_discovery_inputs(
        observations,
        current_candidates,
        training_ids,
        held_out_ids,
        independent_test_ids,
        threshold_array,
    )
    threshold_values = tuple(float(value) for value in threshold_array)
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

    current_result: DiscoveryResult | None = None
    fixed_validation_error: float | None = None
    fixed_training_fit: _TrainingFit | None = None
    rounds: list[dict[str, Any]] = []
    admitted_candidates: list[str] = []
    complexity_layers = sorted(
        {_complexity(candidate) for candidate in available_candidates}
    )
    for round_index, layer in enumerate(
        complexity_layers[:max_rounds], start=1
    ):
        layer_candidates = [
            candidate
            for candidate in available_candidates
            if _complexity(candidate) == layer
        ]
        ranking_fit = _fit_training_only(
            trajectories_by_id,
            training_ids,
            current_candidates,
            threshold_values,
        )
        if fixed_training_fit is None:
            fixed_training_fit = ranking_fit
        residuals = _residual_blocks(
            trajectories_by_id,
            training_ids,
            current_candidates,
            layer_candidates,
            ranking_fit.coefficients,
        )
        ranked = rank_expansions(
            residuals,
            base_candidates=current_candidates,
            expansion_candidates=layer_candidates,
        )

        # Validation starts only after the complete residual ranking exists.
        if current_result is None:
            current_result = discover_closed(
                trajectories=observations,
                candidates=current_candidates,
                target_dimension=target_dimension,
                train_ids=training_ids,
                validation_ids=held_out_ids,
                thresholds=threshold_values,
            )
            fixed_validation_error = current_result.validation_error

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
                target_dimension=target_dimension,
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

        ranking_support = tuple(
            current_candidates[index].name
            for index in ranking_fit.support
            if index < len(current_candidates)
        )
        rounds.append(
            {
                "round": round_index,
                "stage_name": "training-only residual screening",
                "complexity_layer": layer,
                "residual_condition_ids": training_ids,
                "stability_thresholds": {
                    "minimum_absolute_correlation": _MIN_ABSOLUTE_CORRELATION,
                    "minimum_sign_consistency": _MIN_SIGN_CONSISTENCY,
                    "minimum_supporting_condition_count": (
                        len(training_ids) // 2 + 1
                    ),
                },
                "residual_base_fit": {
                    "selection_data": "training_conditions_only",
                    "selection_rule": _RESIDUAL_FIT_SELECTION_RULE,
                    "selected_threshold": ranking_fit.selected_threshold,
                    "training_error": ranking_fit.error,
                    "support": ranking_support,
                },
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
                "validation_admission": {
                    "stage_name": "finite-pool validation admission",
                    "condition_ids": held_out_ids,
                    "rule": (
                        "candidate_selected_and_validation_error_reduction_"
                        "exceeds_one_percent_of_current_error"
                    ),
                    "minimum_error_reduction": minimum_reduction,
                    "admitted_candidate": admitted_name,
                },
            }
        )

    if fixed_training_fit is None:
        fixed_training_fit = _fit_training_only(
            trajectories_by_id,
            training_ids,
            current_candidates,
            threshold_values,
        )
    if current_result is None:
        current_result = discover_closed(
            trajectories=observations,
            candidates=current_candidates,
            target_dimension=target_dimension,
            train_ids=training_ids,
            validation_ids=held_out_ids,
            thresholds=threshold_values,
        )
        fixed_validation_error = current_result.validation_error
    if fixed_validation_error is None:
        raise RuntimeError("fixed-library validation error was not evaluated")

    final_training_error = _training_error(
        trajectories_by_id,
        training_ids,
        current_candidates,
        current_result.coefficients,
    )
    test_normalized_rmse = _training_error(
        trajectories_by_id,
        independent_test_ids,
        current_candidates,
        current_result.coefficients,
    )
    diagnostics = {
        **dict(current_result.diagnostics),
        "stage_names": (
            "training-only residual screening",
            "finite-pool validation admission",
            "independent test evaluation",
        ),
        "condition_roles": {
            "residual_screening_and_training": training_ids,
            "finite_pool_validation_admission": held_out_ids,
            "independent_test_evaluation": independent_test_ids,
        },
        "finite_pool": tuple(
            candidate.name for candidate in expansion_candidates
        ),
        "admission_name": "finite-pool validation admission",
        "claim_boundary": (
            "controlled finite-pool evidence; not unrestricted grammar discovery"
        ),
        "test_ids": independent_test_ids,
        "test_normalized_rmse": test_normalized_rmse,
        "test_evaluation_data": "independent_test_conditions_only",
        "fixed_library_validation_error": fixed_validation_error,
        "fixed_library_training_error": fixed_training_fit.error,
        "fixed_residual_fit_selected_threshold": (
            fixed_training_fit.selected_threshold
        ),
        "residual_fit_selection_data": "training_conditions_only",
        "residual_fit_selection_rule": _RESIDUAL_FIT_SELECTION_RULE,
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
