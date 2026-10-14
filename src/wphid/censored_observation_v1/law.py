"""Exact independent clipped-Gaussian sensor moments and atom likelihoods.

The law is conditional on a supplied latent candidate mean and noise scale.
Neither is inferred from observed RMS or borrowed from held-out clean truth.
Weak centering remains first-order; nonlinear rate/flux bias is not corrected.
Shared latent errors or reconstruction correlations need a different qualified
law and cannot be silently replaced by these independent marginal variances.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, ndtr

from wphid.coupled_balance_v1.assembler import WeakSystem
from wphid.coupled_balance_v1.noise import observation_factor


ARCHIVED_BOUNDS = {"T": (1., np.inf), "pO2": (0., np.inf), "pH2O": (0., np.inf),
    "pProducts": (0., np.inf), "stress": (0., np.inf), "carbon": (.02, 1.), "crack": (0., .92)}


@dataclass(frozen=True)
class ClippedMoments:
    observed_mean: np.ndarray
    mean_error: np.ndarray
    variance: np.ndarray
    lower_atom_probability: np.ndarray
    upper_atom_probability: np.ndarray


@dataclass(frozen=True)
class CensoredGaussianLaw:
    sigma: object
    lower: float
    upper: float
    origin: str
    metadata_identity: str
    latent_errors_independent: bool

    def __post_init__(self):
        sigma = np.asarray(self.sigma, dtype=float)
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("supplied noise scales must be finite and nonnegative")
        if np.isnan(self.lower) or np.isnan(self.upper) or self.lower >= self.upper:
            raise ValueError("ordered clipping bounds required")
        if self.origin not in ("saved_synthetic_noise_metadata", "declared_control_noise", "independent_instrument_calibration"):
            raise ValueError("explicit noise-scale provenance required")
        if not isinstance(self.metadata_identity, str) or not self.metadata_identity:
            raise ValueError("noise metadata/assumption identity required")
        if self.latent_errors_independent is not True:
            raise ValueError("shared latent errors cannot be approximated by independent clipped moments")
        object.__setattr__(self, "sigma", sigma)

    @classmethod
    def archived_channel(cls, channel, sigma, *, origin, metadata_identity, latent_errors_independent):
        if channel not in ARCHIVED_BOUNDS:
            raise ValueError("channel has no declared archived clipping rule")
        return cls(sigma, *ARCHIVED_BOUNDS[channel], origin, metadata_identity, latent_errors_independent)

    def _arrays(self, latent_mean):
        mean, sigma = np.broadcast_arrays(np.asarray(latent_mean, dtype=float), self.sigma)
        if not np.all(np.isfinite(mean)):
            raise ValueError("finite supplied latent candidate mean required")
        return mean, sigma

    def moments(self, latent_mean):
        mean, sigma = self._arrays(latent_mean)
        positive = sigma > 0
        denominator = np.where(positive, sigma, 1.)
        a, b = (self.lower-mean)/denominator, (self.upper-mean)/denominator
        lower_probability, upper_probability = ndtr(a), ndtr(-b)
        # Use survival probabilities in the positive tail to avoid 1-1 loss.
        middle = np.where(a > 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))
        pa, pb = np.exp(-.5*a*a)/np.sqrt(2*np.pi), np.exp(-.5*b*b)/np.sqrt(2*np.pi)
        da = 0. if np.isneginf(self.lower) else self.lower-mean
        db = 0. if np.isposinf(self.upper) else self.upper-mean
        apa, bpb = np.zeros_like(mean), np.zeros_like(mean)
        np.multiply(a, pa, out=apa, where=np.isfinite(a))
        np.multiply(b, pb, out=bpb, where=np.isfinite(b))
        bias = da*lower_probability+sigma*(pa-pb)+db*upper_probability
        second_error = da*da*lower_probability+sigma*sigma*(middle+apa-bpb)+db*db*upper_probability
        variance = second_error-bias*bias
        error_scale = np.maximum(second_error, sigma*sigma)
        if np.any(variance < -64*np.finfo(float).eps*error_scale):
            raise ValueError("numerically unresolved clipped variance")
        variance = np.maximum(variance, 0.)
        deterministic = np.clip(mean, self.lower, self.upper)
        lower_mean = 0. if np.isneginf(self.lower) else self.lower*lower_probability
        upper_mean = 0. if np.isposinf(self.upper) else self.upper*upper_probability
        observed_mean = np.where(positive, lower_mean+mean*middle+sigma*(pa-pb)+upper_mean, deterministic)
        bias = np.where(positive, bias, deterministic-mean)
        variance = np.where(positive, variance, 0.)
        lower_probability = np.where(positive, lower_probability, deterministic == self.lower)
        upper_probability = np.where(positive, upper_probability, deterministic == self.upper)
        return ClippedMoments(observed_mean, bias, variance, lower_probability, upper_probability)

    def negative_log_likelihood(self, observed, latent_mean):
        mean, sigma = self._arrays(latent_mean)
        observed = np.broadcast_to(np.asarray(observed, dtype=float), mean.shape)
        if not np.all(np.isfinite(observed)) or np.any(observed < self.lower) or np.any(observed > self.upper):
            raise ValueError("observation outside clipping law")
        denominator = np.where(sigma > 0, sigma, 1.)
        result = .5*((observed-mean)/denominator)**2+np.log(denominator)+.5*np.log(2*np.pi)
        if np.isfinite(self.lower):
            result = np.where(observed == self.lower, -log_ndtr((self.lower-mean)/denominator), result)
        if np.isfinite(self.upper):
            result = np.where(observed == self.upper, -log_ndtr((mean-self.upper)/denominator), result)
        deterministic = np.where(observed == np.clip(mean, self.lower, self.upper), 0., np.inf)
        return np.where(sigma > 0, result, deterministic)


@dataclass(frozen=True)
class RawMoments:
    coordinate_keys: tuple[str, ...]
    mean_error: np.ndarray
    variance: np.ndarray
    metadata_identity: str
    independent_before_reconstruction: bool

    def __post_init__(self):
        n = len(self.coordinate_keys)
        bias, variance = np.asarray(self.mean_error, dtype=float), np.asarray(self.variance, dtype=float)
        if (n < 1 or len(set(self.coordinate_keys)) != n or bias.shape != (n,) or variance.shape != (n,)
                or not np.all(np.isfinite(bias)) or not np.all(np.isfinite(variance)) or np.any(variance < 0)):
            raise ValueError("moments require finite uniquely named raw coordinates and nonnegative variances")
        if self.independent_before_reconstruction is not True or not self.metadata_identity:
            raise ValueError("explicit independent raw-law identity required; shared errors are not discarded")
        object.__setattr__(self, "mean_error", bias)
        object.__setattr__(self, "variance", variance)


def center_weak_balance(balance, linearization, raw_moments):
    """Conditional first-order centering, retaining target/design cross terms.

    b_c=b-J_b mu and G_c,j=G_j-J_Gj mu give r_c=r-J_r(a)mu.
    Input moments must be evaluated at a candidate latent mean, not clean truth.
    The input mean dependence and nonlinear weak bias still need joint handling.
    """
    if linearization.coordinate_keys != raw_moments.coordinate_keys:
        raise ValueError("raw-law and observation-coordinate identities disagree")
    bias = raw_moments.mean_error
    target = balance.target-linearization.target_jacobian @ bias
    matrix = balance.matrix.copy()
    for index, derivative in enumerate(linearization.column_jacobians):
        matrix[:, index] -= np.asarray(derivative @ bias).ravel()
    # Parameter derivatives of the input moments/mixed Jacobians are unavailable.
    # Do not return the uncentered outer derivatives as if they were corrected.
    centered = WeakSystem(np.asarray(target), matrix, {}, balance.row_keys, balance.column_keys)
    return centered, observation_factor(raw_moments.variance)
