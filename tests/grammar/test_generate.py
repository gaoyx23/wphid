import numpy as np

from wphid.grammar.atoms import aging_atoms
from wphid.grammar.generate import generate_candidates


PRESSURE = (1, -1, -2, 0)
PRESSURE_SQUARED = (2, -2, -4, 0)


def _by_name(candidates, name):
    return next(candidate for candidate in candidates if candidate.name == name)


def test_aging_grammar_admits_required_oxygen_interactions_at_their_dimensions():
    atoms = aging_atoms()

    oxygen_candidates = generate_candidates(atoms, PRESSURE)
    stress_oxygen_candidates = generate_candidates(atoms, PRESSURE_SQUARED)

    assert "arr(T;theta=1000)*pO2*Df" in [candidate.name for candidate in oxygen_candidates]
    assert "arr(T;theta=1000)*pO2*sigma" in [candidate.name for candidate in stress_oxygen_candidates]


def test_generation_filters_out_products_with_a_nonmatching_dimension():
    candidates = generate_candidates(aging_atoms(), PRESSURE)

    assert "arr(T;theta=1000)*pO2*sigma" not in [candidate.name for candidate in candidates]
    assert all(candidate.dimension == PRESSURE for candidate in candidates)


def test_generation_canonicalizes_commutative_products_once():
    candidates = generate_candidates(aging_atoms(), PRESSURE)
    required = [candidate for candidate in candidates if candidate.name == "arr(T;theta=1000)*pO2*Df"]

    assert len(required) == 1
    assert required[0].parents == ("arr(T;theta=1000)", "pO2", "Df")


def test_generation_is_deterministic_and_labels_complexity_groups():
    first = generate_candidates(aging_atoms(), PRESSURE_SQUARED)
    second = generate_candidates(aging_atoms(), PRESSURE_SQUARED)
    pressure_candidates = generate_candidates(aging_atoms(), PRESSURE)

    assert [candidate.name for candidate in first] == [candidate.name for candidate in second]
    assert {candidate.group for candidate in first + pressure_candidates} <= {"F", "G2", "G3"}
    assert _by_name(pressure_candidates, "sigma").group == "F"
    assert _by_name(first, "pO2*sigma").group == "G2"
    assert _by_name(first, "arr(T;theta=1000)*pO2*sigma").group == "G3"


def test_generation_order_is_invariant_to_permuted_atom_input():
    """Depending on caller atom order breaks canonical candidate ordering."""
    atoms = aging_atoms()
    baseline = generate_candidates(atoms, PRESSURE)
    permuted = generate_candidates(
        [atoms[index] for index in (3, 0, 4, 1, 2)], PRESSURE
    )

    assert [candidate.name for candidate in permuted] == [
        candidate.name for candidate in baseline
    ]
    assert [candidate.parents for candidate in permuted] == [
        candidate.parents for candidate in baseline
    ]


def test_declared_saturation_transform_is_available_as_an_atom():
    saturation = _by_name(aging_atoms(), "sat(Df)")

    np.testing.assert_allclose(saturation.evaluate({"Df": np.array([0.0, 0.5])}), [1.0, 2.0])


def test_arrhenius_parameter_changes_evaluator_and_canonical_identity():
    """Different Arrhenius physics must not share one support identity."""
    states = {"T": np.array([400.0])}
    arrhenius_1000 = next(
        candidate
        for candidate in aging_atoms(arrhenius_temperature=1000.0)
        if candidate.name.startswith("arr(T;")
    )
    arrhenius_1800 = next(
        candidate
        for candidate in aging_atoms(arrhenius_temperature=1800.0)
        if candidate.name.startswith("arr(T;")
    )

    assert arrhenius_1000.name == "arr(T;theta=1000)"
    assert arrhenius_1800.name == "arr(T;theta=1800)"
    assert arrhenius_1000.name != arrhenius_1800.name
    np.testing.assert_allclose(arrhenius_1000.evaluate(states), [np.exp(-2.5)])
    np.testing.assert_allclose(arrhenius_1800.evaluate(states), [np.exp(-4.5)])


def test_generated_candidates_preserve_arrhenius_parameter_identity():
    names_1000 = {
        candidate.name
        for candidate in generate_candidates(
            aging_atoms(arrhenius_temperature=1000.0), PRESSURE
        )
    }
    names_1800 = {
        candidate.name
        for candidate in generate_candidates(
            aging_atoms(arrhenius_temperature=1800.0), PRESSURE
        )
    }

    assert "arr(T;theta=1000)*pO2*Df" in names_1000
    assert "arr(T;theta=1800)*pO2*Df" in names_1800
    assert "arr(T;theta=1800)*pO2*Df" not in names_1000
