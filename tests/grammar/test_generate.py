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

    assert "arr(T)*pO2*Df" in [candidate.name for candidate in oxygen_candidates]
    assert "arr(T)*pO2*sigma" in [candidate.name for candidate in stress_oxygen_candidates]


def test_generation_filters_out_products_with_a_nonmatching_dimension():
    candidates = generate_candidates(aging_atoms(), PRESSURE)

    assert "arr(T)*pO2*sigma" not in [candidate.name for candidate in candidates]
    assert all(candidate.dimension == PRESSURE for candidate in candidates)


def test_generation_canonicalizes_commutative_products_once():
    candidates = generate_candidates(aging_atoms(), PRESSURE)
    required = [candidate for candidate in candidates if candidate.name == "arr(T)*pO2*Df"]

    assert len(required) == 1
    assert required[0].parents == ("arr(T)", "pO2", "Df")


def test_generation_is_deterministic_and_labels_complexity_groups():
    first = generate_candidates(aging_atoms(), PRESSURE_SQUARED)
    second = generate_candidates(aging_atoms(), PRESSURE_SQUARED)
    pressure_candidates = generate_candidates(aging_atoms(), PRESSURE)

    assert [candidate.name for candidate in first] == [candidate.name for candidate in second]
    assert {candidate.group for candidate in first + pressure_candidates} <= {"F", "G2", "G3"}
    assert _by_name(pressure_candidates, "sigma").group == "F"
    assert _by_name(first, "pO2*sigma").group == "G2"
    assert _by_name(first, "arr(T)*pO2*sigma").group == "G3"


def test_declared_saturation_transform_is_available_as_an_atom():
    saturation = _by_name(aging_atoms(), "sat(Df)")

    np.testing.assert_allclose(saturation.evaluate({"Df": np.array([0.0, 0.5])}), [1.0, 2.0])
