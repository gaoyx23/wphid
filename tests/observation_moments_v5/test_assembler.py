"""Small synthetic weak arrays; no scientific package loading."""

import numpy as np
from scipy.sparse import csr_matrix

from wphid.final_candidate_v1.problem import WeakCondition, paper_families
from wphid.final_candidate_v1.solver import Candidate, Mechanism
from wphid.observation_moments_v5.weak_assembler import WeakMomentAssembler
from wphid.observation_moments_v5.jets import corrected_inner


def fixture():
    rng = np.random.default_rng(42)
    operators = tuple(csr_matrix(rng.normal(size=(7, 11))) for _ in range(4))
    states = (np.linspace(900., 1300., 11), np.linspace(.4, 1.4, 11), np.linspace(.1, .7, 11))
    bases = tuple(np.column_stack([o @ state for o in operators[:3]]) for state in states)
    targets = tuple(operators[3] @ state + np.linspace(.1, .2, 7) for state in states)
    condition = WeakCondition('artificial', bases, targets, states, operators[2])
    assembler = WeakMomentAssembler((condition,), operators, ((.2, .001, .002),))
    family = next(f for f in paper_families() if f.name == 'arrhenius_oxygen_damage')
    return assembler, family, condition


def test_precontracted_reaction_matches_entire_product_formula():
    assembler, family, condition = fixture()
    result = assembler.response(assembler.blocks[0], family, [7000., .8, 1.3])
    reaction = assembler.reaction_jet(condition.states, family, [7000., .8, 1.3], 1.)
    anchor_family, parameters = assembler.anchor_spec[-1][1:]
    reference = anchor_family.values((np.array([1000.]), np.array([1.]), np.array([.5])), parameters)[0]
    anchor = assembler.reaction_jet(condition.states, anchor_family, parameters, 1/reference)
    expected = corrected_inner(anchor, reaction, assembler.blocks[0]['variance'])
    np.testing.assert_allclose(result[-1], expected, rtol=1e-12, atol=1e-15)


def test_null_and_partial_models_have_identical_common_endpoint():
    assembler, family, _ = fixture()
    null = Candidate()
    partial = Candidate((Mechanism('reaction', family, (0, 0, 1)),))
    assert assembler.matrix(null, []).shape == (75, 9)
    assert assembler.matrix(partial, [7000., .8, 1.3, 8000., 1.1, 1.5]).shape == (75, 12)
    assert len(assembler.row_ids) == 75
    assert len(set(assembler.row_ids)) == 75
    first = assembler.criterion().fingerprint
    assembler.matrix(partial, [6000., .5, 1., 9000., 1.4, 2.])
    assert assembler.criterion().fingerprint == first


def test_multimechanism_missing_destinations_preserve_physical_layout():
    assembler, family, _ = fixture()
    other = next(f for f in paper_families() if f.name == 'damage_power')
    candidate = Candidate((Mechanism('reaction', family, (0, None, 1)),
                           Mechanism('damage', other, (None, 0, 0))))
    matrix = assembler.matrix(candidate, candidate.center)
    assert matrix.shape == (75, 13)
    values = np.arange(13.)
    physical = assembler.physical_coefficients(candidate, values)
    assert [len(c) for c in physical] == [4, 4, 5]
    np.testing.assert_array_equal(np.concatenate(physical), values)
