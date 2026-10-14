"""Weak-form projection utilities."""

from wphid.weak.time import project_time_weak

__all__ = ["project_time_weak"]
from wphid.weak.axisymmetric import (
    AxisymmetricWeakBalance,
    axisymmetric_diffusion_weak_balance,
    axisymmetric_integral,
    axisymmetric_window_slices,
)
from wphid.weak.axisymmetric_library import project_axisymmetric_pde_weak
from wphid.weak.frustum import FrustumWall, frustum_volume_integral, project_frustum_pde_weak
from wphid.weak.bell_shell import (
    bell_shell_volume_integral,
    project_bell_shell_directional_flux_pde_weak,
    project_bell_shell_flux_pde_weak,
    project_bell_shell_flux_pde_weak_with_inner_exchange,
    project_bell_shell_pde_weak,
)

__all__ = [
    "AxisymmetricWeakBalance",
    "axisymmetric_diffusion_weak_balance",
    "axisymmetric_integral",
    "axisymmetric_window_slices",
    "project_axisymmetric_pde_weak",
    "FrustumWall",
    "frustum_volume_integral",
    "project_frustum_pde_weak",
    "bell_shell_volume_integral",
    "project_bell_shell_directional_flux_pde_weak",
    "project_bell_shell_flux_pde_weak",
    "project_bell_shell_flux_pde_weak_with_inner_exchange",
    "project_bell_shell_pde_weak",
]
