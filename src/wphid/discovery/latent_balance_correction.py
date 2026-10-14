"""One linearized minimum-noise state correction, not a latent-state fit.

Conditioned on fixed parameters and original weak equations. Measurements
remain immutable. Correction energy uses the original full noise Jacobian;
re-evaluation of the nonlinear balance checks the linearization, not accuracy.
"""
import torch


def packet_correction(model, kinetics, coefficients, packet):
    with torch.no_grad():
        temperature, oxygen, damage = packet["states"]
        theta, m, q = kinetics[model.group_index].unbind(dim=1)
        rate = torch.exp(-theta[:, None]/temperature)*oxygen**m[:, None]*(1-damage)**q[:, None]
        derivatives = (rate*theta[:, None]/temperature**2,
                       rate*m[:, None]/oxygen, -rate*q[:, None]/(1-damage))
        mass = packet["mass"]
        residual = (packet["targets"]-torch.einsum("ewk,ek->ew", packet["bases"], coefficients[:, :3])
                    -coefficients[:, 3, None]*(rate@mass.T))
        scale = packet["row_scale"].reshape(-1)
        jacobians = []
        for field in range(3):
            rows = []
            for equation in range(3):
                derivative = -coefficients[equation, 3]*mass*derivatives[field][equation][None, :]
                if equation == field:
                    derivative = (derivative-packet["time"]-coefficients[equation, 0]*packet["axial"]
                                  -coefficients[equation, 1]*packet["radial"]-coefficients[equation, 2]*mass)
                rows.append(derivative)
            jacobians.append(torch.cat(rows)*scale[:, None]*packet["noise"][field])
        covariance = sum(j@j.T for j in jacobians)
        covariance = (covariance+covariance.T)/2
        covariance += model.relative_jitter*torch.mean(torch.diag(covariance))*torch.eye(
            len(covariance), dtype=model.dtype, device=model.device)
        factor = torch.linalg.cholesky(covariance)
        scaled = residual.reshape(-1)*scale
        dual = torch.cholesky_solve(scaled[:, None], factor).flatten()
        eta = torch.stack([-j.T@dual for j in jacobians])
        delta = eta*packet["noise"][:, None]
        corrected_states = packet["states"]+delta
        valid = bool(torch.all(corrected_states[:2] > 0) and torch.all(corrected_states[2] < 1))
        linear_left = scaled + sum(j@e for j, e in zip(jacobians, eta))
        quadratic = torch.dot(scaled, dual)
        energy = torch.sum(eta**2)
        row = {"correction_energy": float(energy.cpu()), "original_quadratic": float(quadratic.cpu()),
               "linear_balance_remaining_relative": float((torch.linalg.norm(linear_left)/torch.linalg.norm(scaled).clamp_min(1e-30)).cpu()),
               "correction_rms_in_sigma_by_field": torch.sqrt(torch.mean(eta**2, dim=1)).cpu().tolist(),
               "correction_max_in_sigma_by_field": torch.max(torch.abs(eta), dim=1).values.cpu().tolist(),
               "function_domain_preserved": valid}
        if valid:
            corrected = {**packet, "states": corrected_states,
                "bases": packet["bases"]+torch.stack([delta@packet[k].T for k in ("axial", "radial", "mass")], dim=-1),
                "targets": packet["targets"]-delta@packet["time"].T}
            t, o, d = corrected_states
            updated_rate = torch.exp(-theta[:, None]/t)*o**m[:, None]*(1-d)**q[:, None]
            updated_residual = (corrected["targets"]-torch.einsum("ewk,ek->ew", corrected["bases"], coefficients[:, :3])
                                -coefficients[:, 3, None]*(updated_rate@mass.T))
            whitened = torch.linalg.solve_triangular(factor, (updated_residual.reshape(-1)*scale)[:, None], upper=False)
            row["nonlinear_remaining_quadratic_under_original_covariance"] = float(torch.sum(whitened**2).cpu())
            row["nonlinear_remaining_whitened_norm_relative"] = float(torch.sqrt(torch.sum(whitened**2)/quadratic.clamp_min(1e-30)).cpu())
        return row


def correction_summary(model, vector):
    kinetics, coefficients = model.unpack(model.tensor(vector))
    packets = [packet_correction(model, kinetics, coefficients, packet) for packet in model.packets]
    return {"packets": packets, "correction_energy": sum(p["correction_energy"] for p in packets),
            "original_quadratic": sum(p["original_quadratic"] for p in packets),
            "all_function_domains_preserved": all(p["function_domain_preserved"] for p in packets),
            "claim_boundary": "Minimum-energy linearized correction with numerical jitter; not denoising certification, nonlinear constrained optimization, parameter estimation or confidence coverage."}
