"""Truth-blind numerical conditioning diagnostics for candidate libraries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def diagnose_mechanism_information(base: np.ndarray, atom: np.ndarray) -> dict[str, object]:
    """Measure a mechanism's independent linear information, without targets.

    The residual of a unit-norm atom after projection on the base columns
    distinguishes large column magnitude from genuinely new information.
    Evaluate both pooled and individual-condition designs before interpreting
    fluctuations in individual-condition coefficients. This is conditional on
    the supplied atom; it is NOT an uncertainty interval for its inner parameters.
    """
    base = np.asarray(base, dtype=float)
    atom = np.asarray(atom, dtype=float)
    if (base.ndim != 2 or atom.ndim != 1 or base.shape[0] != atom.size
            or not atom.size or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(atom))):
        raise ValueError("a finite nonempty atom must match a finite base matrix")
    norm = float(np.linalg.norm(atom))
    base_norms = np.linalg.norm(base, axis=0)
    active = base_norms > 0.0
    normalized_base = base[:, active] / base_norms[active]
    if norm == 0.0:
        return {"atom_is_zero": True, "base_rank": int(np.linalg.matrix_rank(normalized_base))
                if normalized_base.shape[1] else 0,
                "independent_energy_fraction": 0.0,
                "conditional_amplitude_noise_gain": None,
                "inner_parameter_uncertainty_established": False}
    unit = atom / norm
    coefficient, _, rank, _ = np.linalg.lstsq(normalized_base, unit, rcond=None)
    residual = unit - normalized_base @ coefficient
    fraction = float(np.clip(residual @ residual, 0.0, 1.0))
    # ||residual(atom)||^-1 is the noise-to-amplitude gain for a fixed atom
    # under spherical additive target noise. Shared state noise violates that
    # assumption; report a sensitivity, never a calibrated standard error.
    gain = 1.0 / (norm * np.sqrt(fraction)) if fraction > np.finfo(float).eps else None
    return {"atom_is_zero": False, "base_rank": int(rank),
            "independent_energy_fraction": fraction,
            "conditional_amplitude_noise_gain": float(gain) if gain is not None else None,
            "inner_parameter_uncertainty_established": False}


def diagnose_library_conditioning(
    matrix: np.ndarray,
    candidate_names: Sequence[str],
    *,
    center_columns: bool = False,
    near_duplicate_cosine: float = 0.999,
    relative_rank_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Report ill-conditioning without using labels or deleting candidates.

    Columns are optionally centred and then normalized by their Euclidean norm
    only for diagnostics.  Returned candidate groups are warnings: numerical
    similarity alone does not authorize choosing one physical mechanism over
    another.
    """

    values = np.asarray(matrix, dtype=float)
    names = tuple(candidate_names)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if values.shape[1] != len(names):
        raise ValueError("candidate_names must align with matrix columns")
    if len(set(names)) != len(names):
        raise ValueError("candidate_names must be unique")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")
    if not 0.0 <= near_duplicate_cosine <= 1.0:
        raise ValueError("near_duplicate_cosine must lie in [0, 1]")
    if not 0.0 < relative_rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie in (0, 1)")

    diagnostic = np.array(values, copy=True)
    if center_columns:
        diagnostic -= np.mean(diagnostic, axis=0, keepdims=True)
    norms = np.linalg.norm(diagnostic, axis=0)
    active = np.flatnonzero(norms > np.finfo(float).eps)
    zero = np.flatnonzero(norms <= np.finfo(float).eps)
    normalized = np.zeros_like(diagnostic)
    if active.size:
        normalized[:, active] = diagnostic[:, active] / norms[active]
        singular = np.linalg.svd(normalized[:, active], compute_uv=False)
        largest = float(singular[0])
        effective_rank = int(np.sum(singular > relative_rank_tolerance * largest))
        condition = (
            float(largest / singular[-1])
            if singular[-1] > np.finfo(float).eps
            else float("inf")
        )
    else:
        singular = np.empty(0, dtype=float)
        effective_rank = 0
        condition = float("inf")

    cosine = normalized.T @ normalized
    pairs = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            absolute = float(abs(cosine[left, right]))
            if left in active and right in active and absolute >= near_duplicate_cosine:
                pairs.append({
                    "left": names[left],
                    "right": names[right],
                    "absolute_cosine": absolute,
                })
    pairs.sort(key=lambda row: (-row["absolute_cosine"], row["left"], row["right"]))
    return {
        "row_count": int(values.shape[0]),
        "column_count": int(values.shape[1]),
        "center_columns": bool(center_columns),
        "near_duplicate_cosine_threshold": float(near_duplicate_cosine),
        "relative_rank_tolerance": float(relative_rank_tolerance),
        "zero_or_constant_after_transform": tuple(names[index] for index in zero),
        "effective_rank": effective_rank,
        "normalized_condition_number": condition,
        "normalized_singular_values": singular.tolist(),
        "near_duplicate_pairs": pairs,
        "automatic_deletion_authorized": False,
    }
