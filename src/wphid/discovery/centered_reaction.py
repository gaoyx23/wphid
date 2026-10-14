"""Exact active-reaction coordinates, not a new model or parameter prior."""
import numpy as np


def log_features(states):
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("nonempty finite three-field array required")
    temperature, oxygen, damage = values
    if np.any(temperature <= 0) or np.any(oxygen <= 0) or np.any(damage >= 1):
        raise ValueError("T/O positive and D<1 required")
    return np.vstack((1/temperature, np.log(oxygen), np.log1p(-damage)))


def observed_pivot(states):
    """One fixed pivot from the declared observation pool, without truth."""
    return log_features(states).mean(axis=1)


def encode_strength(amplitude, parameters, pivot):
    parameters, pivot = np.asarray(parameters), np.asarray(pivot)
    if parameters.shape != (3,) or pivot.shape != (3,) or not np.all(np.isfinite(np.r_[parameters, pivot, amplitude])) or amplitude == 0:
        raise ValueError("finite three-parameter active reaction and nonzero amplitude required")
    theta, m, q = parameters
    return float(np.sign(amplitude)), float(np.log(abs(amplitude))-theta*pivot[0]+m*pivot[1]+q*pivot[2])


def decode_amplitude(sign, log_strength, parameters, pivot):
    if sign not in (-1, 1):
        raise ValueError("active amplitude sign must be +1/-1")
    theta, m, q = np.asarray(parameters)
    return float(sign*np.exp(log_strength+theta*pivot[0]-m*pivot[1]-q*pivot[2]))


def centered_action(states, sign, log_strength, parameters, pivot):
    if sign not in (-1, 1):
        raise ValueError("active amplitude sign must be +1/-1")
    theta, m, q = np.asarray(parameters)
    centered = log_features(states)-np.asarray(pivot)[:, None]
    return sign*np.exp(log_strength-theta*centered[0]+m*centered[1]+q*centered[2])
