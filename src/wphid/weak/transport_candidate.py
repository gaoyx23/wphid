"""Rich-observation weak operator sharing the forward transport syntax.

Requires measured/reconstructed volumetric fields with explicit provenance.
This operator assembles equations, not a calibrated selector. Constitutive
parameters and reaction forcing supplied here are declared known quantities.
"""
import numpy as np

from ..transport_mechanisms import constitutive_factor, modulation_fraction
from .bell_shell import project_bell_shell_directional_flux_pde_weak


def project_transport_candidate(*, candidate, transport, mechanics, species,
                                observed_fields, field_provenance, t, s, x, z, r,
                                forcing, window_shape, stride_shape):
    names = {"T", "stress", "carbon", "crack", species}
    if not names.issubset(observed_fields) or not names.issubset(field_provenance):
        raise ValueError("rich observed fields and their provenance are required")
    if any(field_provenance[n] not in ("measured", "declared_reconstruction") for n in names):
        raise ValueError("hidden truth fields cannot be used as observations")
    diffusivity_name = {"pO2": "oxygen_diffusivity_closed", "pH2O": "steam_diffusivity_closed",
                        "pProducts": "product_diffusivity_closed"}.get(species)
    if diffusivity_name is None:
        raise ValueError("unsupported gas species")
    configured = candidate.configure_transport(transport)
    pressure = np.asarray(observed_fields[species], dtype=float)
    shape = (len(t), len(s), len(x))
    if any(np.asarray(observed_fields[n]).shape != shape for n in names):
        raise ValueError("observations must be aligned volumetric time-space fields")
    fraction = modulation_fraction(candidate.modulation, temperature=observed_fields["T"],
        stress=observed_fields["stress"], carbon=observed_fields["carbon"],
        crack=observed_fields["crack"], transport=configured, mechanics=mechanics)
    gain = 1. + (configured.open_to_closed_diffusivity_ratio - 1.) * fraction
    diffusivity = getattr(configured, diffusivity_name) * gain
    metric = np.sqrt(np.gradient(z, s, axis=0, edge_order=2)**2
                     + np.gradient(r, s, axis=0, edge_order=2)**2)
    grad_s = np.gradient(pressure, s, axis=1, edge_order=2) / metric[None, :, :]
    grad_x = np.gradient(pressure, x, axis=2, edge_order=2)
    d_s = diffusivity * configured.meridional_to_through_diffusivity_ratio
    j_s_linear, j_x_linear = -d_s * grad_s, -diffusivity * grad_x
    # The limiter is positive even at zero gradient; avoid dividing J by J_linear.
    factor_s = constitutive_factor(j_s_linear, configured.forchheimer_inverse_flux_scale)
    factor_x = constitutive_factor(j_x_linear, configured.forchheimer_inverse_flux_scale)
    matrix, target = project_bell_shell_directional_flux_pde_weak(
        t=t, s=s, x=x, z=z, r=r, u=pressure, forcing=forcing,
        meridional_diffusivity_features=(d_s * factor_s)[..., None],
        thickness_diffusivity_features=(diffusivity * factor_x)[..., None],
        reaction_features=np.empty(shape + (0,)), window_shape=window_shape,
        stride_shape=stride_shape,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "matrix": matrix, "target": target,
        "residual": target - np.sum(matrix, axis=1),
        "known_parameters_not_estimated": True,
        "spatial_discretization": "observation_gradient_weak_not_forward_finite_volume",
    }
