"""Local source-coupling diagnostics; this is not an adopted estimator."""

from .geometry import (
    InformationDiagnostic,
    independent_calibration_information,
    nested_nuisance_spaces,
    profile_information,
    weak_source_columns,
)

__all__ = [
    "InformationDiagnostic", "independent_calibration_information",
    "nested_nuisance_spaces", "profile_information", "weak_source_columns",
]
