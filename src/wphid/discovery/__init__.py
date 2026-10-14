"""Residual-driven, physics-grammar expansion."""

from wphid.discovery.residual import (
    ExpansionScore,
    ResidualBlock,
    discover_with_aic_expansion,
    discover_with_expansion,
    rank_expansions,
)
from wphid.discovery.profiled import (
    ProfiledFit,
    fixed_parameter_profile,
    profiled_nonlinear_fit,
)
from wphid.discovery.network_profiled import (
    MultiMechanismProfiledFit,
    NetworkProfiledFit,
    PartitionedMechanismProfiledFit,
    profiled_multi_mechanism_fit,
    profiled_partitioned_mechanism_fit,
    profiled_shared_mechanism_fit,
)
from wphid.discovery.adequacy import (
    absolute_adequacy_decision,
    maximum_reference_bound,
    one_sided_prediction_upper_bound,
)
from wphid.discovery.conditioning import diagnose_library_conditioning

__all__ = [
    "ExpansionScore",
    "ResidualBlock",
    "discover_with_aic_expansion",
    "discover_with_expansion",
    "rank_expansions",
    "ProfiledFit",
    "fixed_parameter_profile",
    "profiled_nonlinear_fit",
    "NetworkProfiledFit",
    "MultiMechanismProfiledFit",
    "PartitionedMechanismProfiledFit",
    "profiled_shared_mechanism_fit",
    "profiled_multi_mechanism_fit",
    "profiled_partitioned_mechanism_fit",
    "absolute_adequacy_decision",
    "maximum_reference_bound",
    "one_sided_prediction_upper_bound",
    "diagnose_library_conditioning",
]
