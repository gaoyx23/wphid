import numpy as np
import pytest

from wphid.discovery.residual import (
    ResidualBlock,
    discover_with_expansion,
    rank_expansions,
)
from wphid.grammar.atoms import PRESSURE, aging_atoms
from wphid.grammar.generate import generate_candidates
from wphid.types import Candidate, Trajectory


TRUE_INTERACTION = "arr(T;theta=1800)*pO2*Df"


def _candidate(name: str, group: str) -> Candidate:
    return Candidate(
        name=name,
        evaluate=lambda states: np.asarray(states[name], dtype=float),
        dimension=PRESSURE,
        group=group,
        parents=tuple(name.split("*")),
        coefficient_dimension=(0, 0, 0, 0),
    )


def test_ranking_requires_stable_cross_condition_weak_residual_evidence():
    """A distractor supported in only one condition must not be expandable."""
    rng = np.random.default_rng(20260729)
    generated = generate_candidates(aging_atoms(1800.0), PRESSURE)
    true_candidate = next(
        candidate for candidate in generated if candidate.name == TRUE_INTERACTION
    )
    one_condition_candidate = _candidate("one-condition-only", "G3")
    base_candidates = [
        candidate
        for candidate in generated
        if candidate.name == "arr(T;theta=1800)*pO2"
    ]

    residuals_by_condition = {}
    for condition_index, condition_id in enumerate(("c0", "c1", "c2")):
        states = {
            "T": rng.uniform(320.0, 430.0, 240),
            "pO2": rng.uniform(0.1, 0.9, 240),
            "Df": rng.uniform(0.02, 0.5, 240),
        }
        true_column = true_candidate.evaluate(states)
        residual = 0.45 * true_column + rng.normal(
            scale=0.03 * np.std(true_column), size=true_column.size
        )
        one_condition_column = (
            residual + rng.normal(scale=0.01 * np.std(residual), size=residual.size)
            if condition_index == 0
            else rng.normal(size=residual.size)
        )
        residuals_by_condition[condition_id] = ResidualBlock(
            residual=residual,
            weak_columns={
                TRUE_INTERACTION: true_column,
                one_condition_candidate.name: one_condition_column,
            },
        )

    ranked = rank_expansions(
        residuals_by_condition,
        base_candidates=base_candidates,
        expansion_candidates=[one_condition_candidate, true_candidate],
    )

    assert ranked[0].candidate.name == TRUE_INTERACTION
    assert ranked[0].stable
    one_condition_score = next(
        score
        for score in ranked
        if score.candidate.name == one_condition_candidate.name
    )
    assert not one_condition_score.stable
    assert one_condition_score.median_absolute_correlation < 0.25


def test_expansion_rejects_overlap_with_independent_test_conditions():
    """A condition reused for admission and test would leak into evaluation."""
    time = np.linspace(0.0, 1.0, 21)
    trajectories = [
        Trajectory(
            t=time,
            states={
                "Do": time**2,
                "base": np.ones_like(time),
            },
            condition_id=condition_id,
        )
        for condition_id in ("train-a", "train-b", "validation")
    ]

    with pytest.raises(ValueError, match="pairwise disjoint"):
        discover_with_expansion(
            trajectories=trajectories,
            base_candidates=[_candidate("base", "F")],
            expansion_candidates=[],
            train_ids=("train-a", "train-b"),
            validation_ids=("validation",),
            test_ids=("validation",),
            target_dimension=PRESSURE,
            thresholds=(0.0,),
        )
