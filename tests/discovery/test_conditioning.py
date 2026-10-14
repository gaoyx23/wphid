import numpy as np
import pytest

from wphid.discovery.conditioning import diagnose_library_conditioning


def test_conditioning_flags_duplicate_columns_without_deleting_them():
    coordinate = np.linspace(-1.0, 1.0, 101)
    matrix = np.column_stack((coordinate, 3.0 * coordinate, coordinate**2))
    report = diagnose_library_conditioning(matrix, ("x", "3x", "x2"))

    assert report["effective_rank"] == 2
    assert report["automatic_deletion_authorized"] is False
    assert report["near_duplicate_pairs"] == [
        {"left": "x", "right": "3x", "absolute_cosine": pytest.approx(1.0)}
    ]


def test_conditioning_is_invariant_to_positive_column_scale():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(80, 3))
    first = diagnose_library_conditioning(matrix, ("a", "b", "c"))
    second = diagnose_library_conditioning(
        matrix * np.asarray([1.0e-6, 4.0, 2.0e3]), ("a", "b", "c")
    )

    assert second["normalized_singular_values"] == pytest.approx(
        first["normalized_singular_values"]
    )
    assert second["normalized_condition_number"] == pytest.approx(
        first["normalized_condition_number"]
    )


def test_centering_reports_constant_column_as_uninformative():
    coordinate = np.linspace(0.0, 1.0, 50)
    matrix = np.column_stack((np.ones_like(coordinate), coordinate))
    report = diagnose_library_conditioning(
        matrix, ("constant", "ramp"), center_columns=True
    )

    assert report["zero_or_constant_after_transform"] == ("constant",)
    assert report["effective_rank"] == 1


def test_conditioning_rejects_misaligned_names():
    with pytest.raises(ValueError, match="align"):
        diagnose_library_conditioning(np.ones((5, 2)), ("only-one",))
