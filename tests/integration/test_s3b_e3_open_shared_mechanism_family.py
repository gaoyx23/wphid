from experiments.s3b_e3_open_shared_mechanism_family import (
    FAMILIES,
    _point_atom,
)


def test_shared_family_library_contains_null_and_distinct_reductions():
    assert FAMILIES[0] == "null"
    full = _point_atom("arrhenius_oxygen_damage", [7200, .65, 1.25], 1200, 1.2, .3)
    reduced = _point_atom("arrhenius_oxygen", [7200, .65], 1200, 1.2, .3)
    assert full != reduced
