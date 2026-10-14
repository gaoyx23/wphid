"""Known-parameter weak candidate-null sets, not general discovery."""
import numpy as np
from .empirical_noise import whiten_noise


def candidate_null_set(*, residuals, null_models):
    """Each candidate uses its OWN simulated baseline, metric and radius.

    Observations are raw residuals evaluated under each hypothesis. Never
    subtract the unknown observation source's clean residual from them.
    Null models must share observation configuration and physical conditions.
    """
    if not residuals or set(residuals) != set(null_models):
        raise ValueError("all declared candidate residuals and null models required")
    retained, scores = [], {}
    observation_configuration_ids = {model["observation_configuration_id"] for model in null_models.values()}
    if len(observation_configuration_ids) != 1 or None in observation_configuration_ids:
        raise ValueError("candidate null observation configurations must agree")
    for name, model in null_models.items():
        if model.get("candidate_id") != name:
            raise ValueError("candidate null identity mismatch")
        baseline = np.asarray(model["candidate_simulated_clean_baseline"], dtype=float)
        metric = dict(model["noise_metric"])
        bias = np.asarray(metric["center"], dtype=float)
        if baseline.shape != bias.shape or not np.all(np.isfinite(baseline)):
            raise ValueError("finite aligned candidate-simulated baseline required")
        metric["center"] = baseline + bias
        radius = model["calibrated_squared_radius"]
        if radius is None or not np.isfinite(radius) or radius <= 0.:
            raise ValueError("finite positive independently calibrated radius required")
        vector = np.asarray(residuals[name], dtype=float)
        white = whiten_noise(vector[None, :], metric)[0]
        score = float(np.sum(white**2))
        scores[name] = dict(score=score, squared_radius=float(radius),
            inside=score <= radius, null_simulation_source=model["null_simulation_source"])
        if score <= radius: retained.append(name)
    return dict(candidate_set=retained, scores=scores,
        interpretation="empty_model_inadequate" if not retained else "unique_within_declared_fixed_parameter_library" if len(retained)==1 else "candidate_set",
        parameter_recovery=False, engineering_confirmation=False)
