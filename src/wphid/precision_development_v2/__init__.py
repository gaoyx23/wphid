"""Independent precision-development solver; not a qualified final method."""

VERSION = "WPHID_COMPONENT_PROFILE_DEVELOPMENT_V2"

from .factorization import factorize, prepare_profile, profile_components
from .solver import FactoredFit, fit_factored_candidate

__all__ = ["VERSION", "factorize", "prepare_profile", "profile_components",
           "FactoredFit", "fit_factored_candidate"]
