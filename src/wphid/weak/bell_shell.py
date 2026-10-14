"""Space-time weak operators on an axisymmetric curved thin shell."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .axisymmetric import axisymmetric_window_slices


def bell_shell_volume_integral(values, t, s, x, z, r) -> float:
    """Integrate with the shell measure ``r*h_s ds dx dt`` (without ``2*pi``)."""

    time, meridional, depth, axial, radius = _coordinates(t, s, x, z, r)
    field = np.asarray(values, dtype=float)
    shape = (time.size, meridional.size, depth.size)
    if field.shape != shape or not np.all(np.isfinite(field)):
        raise ValueError(f"values must be finite with shape {shape}")
    jacobian, _ = _metrics(meridional, axial, radius)
    return _integrate(field * jacobian[None, :, :], time, meridional, depth)


def project_bell_shell_pde_weak(
    *,
    t,
    s,
    x,
    z,
    r,
    u,
    forcing,
    reaction_features,
    window_shape,
    stride_shape,
):
    """Assemble meridional diffusion, thickness diffusion and reaction columns.

    The physical wall is represented by ``(z(s,x), r(s,x))``.  Compact test
    functions move both diffusion derivatives away from the observed field.
    """

    time, meridional, depth, axial, radius = _coordinates(t, s, x, z, r)
    shape = (time.size, meridional.size, depth.size)
    values = _field(u, shape, "u")
    source = _field(forcing, shape, "forcing")
    features = np.asarray(reaction_features, dtype=float)
    if (
        features.ndim != 4
        or features.shape[:3] != shape
        or not np.all(np.isfinite(features))
    ):
        raise ValueError(
            "reaction_features must be finite with shape (n_t,n_s,n_x,n_terms)"
        )

    rows, targets = [], []
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        ts = time[slices[0]]
        ss = meridional[slices[1]]
        xs = depth[slices[2]]
        local_u = values[slices]
        local_f = source[slices]
        local_g = features[slices]
        local_z = axial[slices[1], slices[2]]
        local_r = radius[slices[1], slices[2]]
        jacobian, h_s = _metrics(ss, local_z, local_r)

        bt, bt_t, _ = _bump(ts)
        bs, bs_s, bs_ss = _bump(ss)
        bx, bx_x, bx_xx = _bump(xs)
        phi = bt[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_t = bt_t[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_s = bt[:, None, None] * bs_s[None, :, None] * bx[None, None, :]
        phi_ss = bt[:, None, None] * bs_ss[None, :, None] * bx[None, None, :]
        phi_x = bt[:, None, None] * bs[None, :, None] * bx_x[None, None, :]
        phi_xx = bt[:, None, None] * bs[None, :, None] * bx_xx[None, None, :]

        q_s_metric = local_r / h_s
        q_s_derivative = np.gradient(q_s_metric, ss, axis=0, edge_order=2)
        q_x_metric = jacobian
        q_x_derivative = np.gradient(q_x_metric, xs, axis=1, edge_order=2)
        meridional_adjoint = (
            q_s_derivative[None, :, :] * phi_s
            + q_s_metric[None, :, :] * phi_ss
        )
        thickness_adjoint = (
            q_x_derivative[None, :, :] * phi_x
            + q_x_metric[None, :, :] * phi_xx
        )
        meridional_column = _integrate(
            local_u * meridional_adjoint, ts, ss, xs
        )
        thickness_column = _integrate(
            local_u * thickness_adjoint, ts, ss, xs
        )
        reaction_columns = [
            _integrate(
                local_g[..., index] * phi * jacobian[None, :, :], ts, ss, xs
            )
            for index in range(features.shape[3])
        ]
        target = -_integrate(
            local_u * phi_t * jacobian[None, :, :], ts, ss, xs
        ) - _integrate(local_f * phi * jacobian[None, :, :], ts, ss, xs)
        rows.append([meridional_column, thickness_column, *reaction_columns])
        targets.append(target)
    return np.asarray(rows), np.asarray(targets)


def project_bell_shell_flux_pde_weak(
    *,
    t,
    s,
    x,
    z,
    r,
    u,
    forcing,
    diffusivity_features,
    reaction_features,
    window_shape,
    stride_shape,
):
    """Project variable diffusion in conservative flux form.

    Each diffusivity feature ``a_j`` contributes the weak column
    ``-integral a_j grad(u) dot grad(phi) dV`` for the operator
    ``div(a_j grad(u))``.  Only one spatial derivative is taken from the
    reconstructed state, while the outer divergence is transferred to the
    compact test function.  Reaction columns and the temporal target use the
    same curved-shell measure as :func:`project_bell_shell_pde_weak`.
    """

    time, meridional, depth, axial, radius = _coordinates(t, s, x, z, r)
    shape = (time.size, meridional.size, depth.size)
    values = _field(u, shape, "u")
    source = _field(forcing, shape, "forcing")
    diffusion = np.asarray(diffusivity_features, dtype=float)
    reactions = np.asarray(reaction_features, dtype=float)
    for name, features in (
        ("diffusivity_features", diffusion),
        ("reaction_features", reactions),
    ):
        if (
            features.ndim != 4
            or features.shape[:3] != shape
            or not np.all(np.isfinite(features))
        ):
            raise ValueError(
                f"{name} must be finite with shape (n_t,n_s,n_x,n_terms)"
            )
    if diffusion.shape[3] < 1:
        raise ValueError("diffusivity_features must contain at least one term")

    rows, targets = [], []
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        ts = time[slices[0]]
        ss = meridional[slices[1]]
        xs = depth[slices[2]]
        local_u = values[slices]
        local_f = source[slices]
        local_a = diffusion[slices]
        local_g = reactions[slices]
        local_z = axial[slices[1], slices[2]]
        local_r = radius[slices[1], slices[2]]
        jacobian, h_s = _metrics(ss, local_z, local_r)

        bt, bt_t, _ = _bump(ts)
        bs, bs_s, _ = _bump(ss)
        bx, bx_x, _ = _bump(xs)
        phi = bt[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_t = bt_t[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_s = bt[:, None, None] * bs_s[None, :, None] * bx[None, None, :]
        phi_x = bt[:, None, None] * bs[None, :, None] * bx_x[None, None, :]

        u_s = np.gradient(local_u, ss, axis=1, edge_order=2)
        u_x = np.gradient(local_u, xs, axis=2, edge_order=2)
        gradient_pairing = (
            u_s * phi_s / h_s[None, :, :] ** 2 + u_x * phi_x
        )
        diffusion_columns = [
            -_integrate(
                local_a[..., index]
                * gradient_pairing
                * jacobian[None, :, :],
                ts,
                ss,
                xs,
            )
            for index in range(diffusion.shape[3])
        ]
        reaction_columns = [
            _integrate(
                local_g[..., index] * phi * jacobian[None, :, :],
                ts,
                ss,
                xs,
            )
            for index in range(reactions.shape[3])
        ]
        target = -_integrate(
            local_u * phi_t * jacobian[None, :, :], ts, ss, xs
        ) - _integrate(local_f * phi * jacobian[None, :, :], ts, ss, xs)
        rows.append([*diffusion_columns, *reaction_columns])
        targets.append(target)
    return np.asarray(rows), np.asarray(targets)


def project_bell_shell_directional_flux_pde_weak(
    *,
    t,
    s,
    x,
    z,
    r,
    u,
    forcing,
    meridional_diffusivity_features,
    thickness_diffusivity_features,
    reaction_features,
    window_shape,
    stride_shape,
):
    """Project tensor-diagonal variable diffusion in conservative weak form.

    Meridional and through-thickness features form separate column blocks, so
    the scalar-isotropic constraint can be tested against orthotropic CMC gas
    transport without changing the curved-shell or reaction measures.
    """

    time, meridional, depth, axial, radius = _coordinates(t, s, x, z, r)
    shape = (time.size, meridional.size, depth.size)
    values = _field(u, shape, "u")
    source = _field(forcing, shape, "forcing")
    meridional_features = np.asarray(
        meridional_diffusivity_features, dtype=float
    )
    thickness_features = np.asarray(
        thickness_diffusivity_features, dtype=float
    )
    reactions = np.asarray(reaction_features, dtype=float)
    for name, features in (
        ("meridional_diffusivity_features", meridional_features),
        ("thickness_diffusivity_features", thickness_features),
        ("reaction_features", reactions),
    ):
        if (
            features.ndim != 4
            or features.shape[:3] != shape
            or not np.all(np.isfinite(features))
        ):
            raise ValueError(
                f"{name} must be finite with shape (n_t,n_s,n_x,n_terms)"
            )
    if meridional_features.shape[3] < 1 or thickness_features.shape[3] < 1:
        raise ValueError("each directional diffusivity block needs at least one term")

    rows, targets = [], []
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        ts = time[slices[0]]
        ss = meridional[slices[1]]
        xs = depth[slices[2]]
        local_u = values[slices]
        local_f = source[slices]
        local_s_features = meridional_features[slices]
        local_x_features = thickness_features[slices]
        local_g = reactions[slices]
        local_z = axial[slices[1], slices[2]]
        local_r = radius[slices[1], slices[2]]
        jacobian, h_s = _metrics(ss, local_z, local_r)

        bt, bt_t, _ = _bump(ts)
        bs, bs_s, _ = _bump(ss)
        bx, bx_x, _ = _bump(xs)
        phi = bt[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_t = bt_t[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_s = bt[:, None, None] * bs_s[None, :, None] * bx[None, None, :]
        phi_x = bt[:, None, None] * bs[None, :, None] * bx_x[None, None, :]

        u_s = np.gradient(local_u, ss, axis=1, edge_order=2)
        u_x = np.gradient(local_u, xs, axis=2, edge_order=2)
        meridional_pairing = u_s * phi_s / h_s[None, :, :] ** 2
        thickness_pairing = u_x * phi_x
        meridional_columns = [
            -_integrate(
                local_s_features[..., index]
                * meridional_pairing
                * jacobian[None, :, :],
                ts,
                ss,
                xs,
            )
            for index in range(meridional_features.shape[3])
        ]
        thickness_columns = [
            -_integrate(
                local_x_features[..., index]
                * thickness_pairing
                * jacobian[None, :, :],
                ts,
                ss,
                xs,
            )
            for index in range(thickness_features.shape[3])
        ]
        reaction_columns = [
            _integrate(
                local_g[..., index] * phi * jacobian[None, :, :],
                ts,
                ss,
                xs,
            )
            for index in range(reactions.shape[3])
        ]
        target = -_integrate(
            local_u * phi_t * jacobian[None, :, :], ts, ss, xs
        ) - _integrate(local_f * phi * jacobian[None, :, :], ts, ss, xs)
        rows.append([
            *meridional_columns,
            *thickness_columns,
            *reaction_columns,
        ])
        targets.append(target)
    return np.asarray(rows), np.asarray(targets)


def project_bell_shell_flux_pde_weak_with_inner_exchange(
    *,
    t,
    s,
    x,
    z,
    r,
    u,
    forcing,
    diffusivity_features,
    inner_exchange_features,
    reaction_features,
    window_shape,
    stride_shape,
    exchange_side="inner",
):
    """Project variable diffusion while retaining the hot-face exchange term.

    The thickness test function equals one at the requested wall surface and
    vanishes smoothly at the opposite surface.  Consequently that boundary
    contribution is not discarded.  ``exchange_side='inner'`` preserves the
    legacy behavior; ``'outer'`` provides the mirrored cold-face balance.
    Each ``inner_exchange_features`` column represents a boundary driving force
    such as ``p_external - p_surface`` and receives a fitted surface-transfer
    coefficient.  Meridional and temporal test functions remain compact, so no
    unreported end-of-nozzle or time-boundary terms enter the balance.

    Columns are returned in the order diffusion, inner exchange, and reaction.
    """

    time, meridional, depth, axial, radius = _coordinates(t, s, x, z, r)
    shape = (time.size, meridional.size, depth.size)
    values = _field(u, shape, "u")
    source = _field(forcing, shape, "forcing")
    diffusion = np.asarray(diffusivity_features, dtype=float)
    exchange = np.asarray(inner_exchange_features, dtype=float)
    reactions = np.asarray(reaction_features, dtype=float)
    for name, features in (
        ("diffusivity_features", diffusion),
        ("reaction_features", reactions),
    ):
        if (
            features.ndim != 4
            or features.shape[:3] != shape
            or not np.all(np.isfinite(features))
        ):
            raise ValueError(
                f"{name} must be finite with shape (n_t,n_s,n_x,n_terms)"
            )
    if (
        exchange.ndim != 3
        or exchange.shape[:2] != shape[:2]
        or not np.all(np.isfinite(exchange))
    ):
        raise ValueError(
            "inner_exchange_features must be finite with shape (n_t,n_s,n_terms)"
        )
    if diffusion.shape[3] < 1:
        raise ValueError("diffusivity_features must contain at least one term")
    if exchange.shape[2] < 1:
        raise ValueError("inner_exchange_features must contain at least one term")
    if exchange_side not in {"inner", "outer"}:
        raise ValueError("exchange_side must be 'inner' or 'outer'")

    rows, targets = [], []
    for slices in axisymmetric_window_slices(shape, window_shape, stride_shape):
        ts = time[slices[0]]
        ss = meridional[slices[1]]
        xs = depth[slices[2]]
        local_u = values[slices]
        local_f = source[slices]
        local_a = diffusion[slices]
        local_h = exchange[slices[0], slices[1], :]
        local_g = reactions[slices]
        local_z = axial[slices[1], slices[2]]
        local_r = radius[slices[1], slices[2]]
        jacobian, h_s = _metrics(ss, local_z, local_r)

        bt, bt_t, _ = _bump(ts)
        bs, bs_s, _ = _bump(ss)
        if exchange_side == "inner":
            bx, bx_x = _inner_anchored_bump(xs)
            boundary_index = 0
        else:
            bx, bx_x = _outer_anchored_bump(xs)
            boundary_index = -1
        phi = bt[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_t = bt_t[:, None, None] * bs[None, :, None] * bx[None, None, :]
        phi_s = bt[:, None, None] * bs_s[None, :, None] * bx[None, None, :]
        phi_x = bt[:, None, None] * bs[None, :, None] * bx_x[None, None, :]

        u_s = np.gradient(local_u, ss, axis=1, edge_order=2)
        u_x = np.gradient(local_u, xs, axis=2, edge_order=2)
        gradient_pairing = (
            u_s * phi_s / h_s[None, :, :] ** 2 + u_x * phi_x
        )
        diffusion_columns = [
            -_integrate(
                local_a[..., index]
                * gradient_pairing
                * jacobian[None, :, :],
                ts,
                ss,
                xs,
            )
            for index in range(diffusion.shape[3])
        ]
        boundary_weight = (
            bt[:, None] * bs[None, :] * jacobian[None, :, boundary_index]
        )
        exchange_columns = [
            _integrate_surface(
                local_h[..., index] * boundary_weight, ts, ss
            )
            for index in range(exchange.shape[2])
        ]
        reaction_columns = [
            _integrate(
                local_g[..., index] * phi * jacobian[None, :, :], ts, ss, xs
            )
            for index in range(reactions.shape[3])
        ]
        target = -_integrate(
            local_u * phi_t * jacobian[None, :, :], ts, ss, xs
        ) - _integrate(local_f * phi * jacobian[None, :, :], ts, ss, xs)
        rows.append([*diffusion_columns, *exchange_columns, *reaction_columns])
        targets.append(target)
    return np.asarray(rows), np.asarray(targets)


def _metrics(s, z, r):
    z_s = np.gradient(z, s, axis=0, edge_order=2)
    r_s = np.gradient(r, s, axis=0, edge_order=2)
    h_s = np.sqrt(z_s**2 + r_s**2)
    if np.any(h_s <= 0.0) or not np.all(np.isfinite(h_s)):
        raise ValueError("shell mapping has a singular meridional metric")
    return r * h_s, h_s


def _bump(coordinate):
    centre = 0.5 * (coordinate[0] + coordinate[-1])
    half = 0.5 * (coordinate[-1] - coordinate[0])
    normalized = (coordinate - centre) / half
    base = 1.0 - normalized**2
    return (
        base**4,
        -8.0 * normalized * base**3 / half,
        -8.0 * base**2 * (1.0 - 7.0 * normalized**2) / half**2,
    )


def _inner_anchored_bump(coordinate):
    span = coordinate[-1] - coordinate[0]
    normalized = (coordinate - coordinate[0]) / span
    base = 1.0 - normalized
    return base**4, -4.0 * base**3 / span


def _outer_anchored_bump(coordinate):
    span = coordinate[-1] - coordinate[0]
    normalized = (coordinate - coordinate[0]) / span
    return normalized**4, 4.0 * normalized**3 / span


def _integrate(values, t, s, x):
    return float(
        simpson(
            simpson(simpson(values, x=x, axis=2), x=s, axis=1),
            x=t,
            axis=0,
        )
    )


def _integrate_surface(values, t, s):
    return float(simpson(simpson(values, x=s, axis=1), x=t, axis=0))


def _coordinates(t, s, x, z, r):
    output = []
    for name, values in (("t", t), ("s", s), ("x", x)):
        coordinate = np.asarray(values, dtype=float)
        if (
            coordinate.ndim != 1
            or coordinate.size < 3
            or not np.all(np.isfinite(coordinate))
            or not np.all(np.diff(coordinate) > 0.0)
        ):
            raise ValueError(f"{name} must be finite, one-dimensional, and increasing")
        output.append(coordinate)
    axial = np.asarray(z, dtype=float)
    radius = np.asarray(r, dtype=float)
    expected = (output[1].size, output[2].size)
    if axial.shape != expected or radius.shape != expected:
        raise ValueError(f"z and r must have shape {expected}")
    if not np.all(np.isfinite(axial)) or not np.all(np.isfinite(radius)):
        raise ValueError("z and r must be finite")
    if np.any(radius <= 0.0):
        raise ValueError("shell radius must be positive")
    return (*output, axial, radius)


def _field(values, shape, name):
    field = np.asarray(values, dtype=float)
    if field.shape != shape or not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return field
