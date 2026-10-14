"""One fixed saved-model carbon balance/local-geometry audit; no refitting.

All fields are previously viewed clean synthetic arrays. This cannot establish
new parameter accuracy, calibration, open discovery, or independent confirmation.
"""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/source_balance_v1/carbon-audit-protocol.json"
KINETIC_KEYS = ("oxygen_activation_energy", "steam_activation_energy",
                "oxygen_exponent", "steam_exponent", "carbon_exponent")
KINETIC_UNITS = np.array([1e5, 1e5, 1., 1., 1.])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def rate_columns(temperature, oxygen, steam, carbon, parameters, gas_constant):
    if np.any(temperature <= 0) or np.any(oxygen < 0) or np.any(steam < 0) or np.any(carbon <= 0):
        raise ValueError("state outside the declared carbon source domain")
    oxygen_activity, steam_activity = oxygen / 2e4, steam / 1e6
    phi_o = (np.exp(-parameters["oxygen_activation_energy"] / (gas_constant*temperature))
             * oxygen_activity**parameters["oxygen_exponent"]
             * carbon**parameters["carbon_exponent"])
    phi_w = (np.exp(-parameters["steam_activation_energy"] / (gas_constant*temperature))
             * steam_activity**parameters["steam_exponent"]
             * carbon**parameters["carbon_exponent"])
    ro = parameters["oxygen_pre_exponential"] * phi_o
    rw = parameters["steam_pre_exponential"] * phi_w
    log_o, log_w = np.zeros_like(oxygen_activity), np.zeros_like(steam_activity)
    np.log(oxygen_activity, out=log_o, where=oxygen_activity > 0)
    np.log(steam_activity, out=log_w, where=steam_activity > 0)
    derivative = np.stack((
        ro * 1e5 / (gas_constant*temperature),
        rw * 1e5 / (gas_constant*temperature), -ro*log_o, -rw*log_w,
        -(ro+rw)*np.log(carbon),
    ), axis=-1)
    return -(ro+rw), np.stack((-phi_o, -phi_w), axis=-1), derivative


def weak_weights(times):
    differences = np.diff(times)
    if len(times) % 2 != 1 or not np.allclose(differences, differences[0], rtol=1e-10, atol=1e-10):
        raise ValueError("fixed composite Simpson rule requires an odd uniform window")
    coordinate = 2*(times-times[0])/(times[-1]-times[0])-1
    test = 1-coordinate**2
    derivative = -4*coordinate/(times[-1]-times[0])
    quadrature = np.ones(len(times))
    quadrature[1:-1:2], quadrature[2:-1:2] = 4., 2.
    quadrature *= differences[0]/3
    return quadrature*test, -quadrature*derivative


def numerical_checks(parameters, gas_constant):
    t = np.arange(17)*6.
    weights, target_weights = weak_weights(t)
    np.testing.assert_allclose(target_weights @ (1-.002*t), weights @ np.full(17, -.002), atol=1e-12)
    states = (np.array([900., 1500., 2100.]), np.array([6000., 2e4, 5e4]),
              np.array([2e5, 7e5, 1.1e6]), np.array([.7, .9, .98]))
    analytic = rate_columns(*states, parameters, gas_constant)[2]
    for index, key in enumerate(KINETIC_KEYS):
        upper, lower = dict(parameters), dict(parameters)
        step = 1e-6
        upper[key] += step*KINETIC_UNITS[index]
        lower[key] -= step*KINETIC_UNITS[index]
        difference = (rate_columns(*states, upper, gas_constant)[0]
                      -rate_columns(*states, lower, gas_constant)[0])/(2*step)
        np.testing.assert_allclose(analytic[:, index], difference, rtol=1e-7, atol=1e-14)
    return 6


def diagnostic_record(derivative, nuisance, diagnose):
    diagnostic = diagnose(derivative, nuisance, rtol=1e-10)
    _, _, right = np.linalg.svd(diagnostic.residual_derivative, full_matrices=False)
    return {
        "kinetic_parameters": list(KINETIC_KEYS), "parameter_units": KINETIC_UNITS.tolist(),
        "kinetic_rank": diagnostic.rank, "kinetic_dimension": diagnostic.n_parameters,
        "nuisance_rank": diagnostic.nuisance_rank,
        "singular_values": diagnostic.singular_values.tolist(),
        "smallest_singular_value": diagnostic.smallest_singular_value,
        "condition_ratio": diagnostic.condition_ratio,
        "weakest_unit_coordinate_direction": dict(zip(KINETIC_KEYS, right[-1].tolist())),
        "metric": "fixed identity on weak rows; correlated/overlapping rows are not independent samples",
        "confidence_or_actual_precision_established": False,
    }


def main():
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the actual algorithm worktree")
    sys.path.insert(0, str(ROOT / "src"))
    import wphid
    from wphid.source_geometry_v6 import profile_information
    import wphid.source_geometry_v6.geometry as geometry
    package, module = Path(wphid.__file__).resolve(), Path(geometry.__file__).resolve()
    if not package.is_relative_to(ROOT / "src") or not module.is_relative_to(ROOT / "src"):
        raise SystemExit("main editable install is forbidden")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    output = (ROOT / protocol["output_root"]).resolve()
    if not output.is_relative_to(ROOT / "artifacts/paper_source_balance_v1") or output.exists():
        raise SystemExit("completed/partial output retained; no repeat or overwrite")
    main_root = Path(protocol["main_root_read_only"]).resolve()
    baseline = json.loads((ROOT / protocol["baseline_manifest"]).read_text(encoding="utf-8"))
    omitted = {item["path"]: item for item in baseline["excluded"]}
    completion_path = ROOT / protocol["prior_input_completion"]
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if not completion["raw_array_qualification_passed"]:
        raise SystemExit("saved array eligibility is not complete")
    if sha(completion_path.parent / "assessment.json") != completion["assessment_sha256"]:
        raise SystemExit("prior array qualification identity mismatch")
    model_path = ROOT / protocol["saved_model"]["path"]
    if sha(model_path) != protocol["saved_model"]["sha256"]:
        raise SystemExit("saved model identity mismatch")
    # Parse the saved artifact but use only its pooled fit and success flag;
    # neither truth_scoring_only nor parameter-error/fold scores choose the model.
    saved = json.loads(model_path.read_text(encoding="utf-8"))["results"]["chemistry"]
    if not saved["success"]:
        raise SystemExit("selected historical pooled fit was not successful")
    parameters = saved["parameters"]
    paths = []
    for item in protocol["inputs"]:
        path = (main_root / item["path"]).resolve()
        if not path.is_relative_to(main_root) or item["path"] not in omitted:
            raise SystemExit("input is not a registered read-only artifact")
        if path.stat().st_size != omitted[item["path"]]["bytes"] or sha(path) != item["sha256"]:
            raise SystemExit("saved input byte/hash mismatch")
        paths.append(path)
    for record in protocol["frozen_dependency_hashes"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise SystemExit("frozen dependency changed")
    identities = [{"path": p, "sha256": sha(ROOT / p)} for p in protocol["source_files"]]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "input-source-freeze.json", {
        "protocol": protocol["version"], "cwd": str(ROOT), "imported_wphid": str(package),
        "imported_diagnostic": str(module), "output_root": str(output),
        "inputs": protocol["inputs"], "saved_model": protocol["saved_model"],
        "pooled_parameters_used": parameters, "source_files": identities,
        "truth_or_error_fields_used_for_model_selection": False,
    })
    try:
        checks = numerical_checks(parameters, protocol["gas_constant"])
        derivatives, nuisances, rows = [], [], []
        for item, path in zip(protocol["inputs"], paths):
            if time.perf_counter()-started > protocol["wall_cap_seconds"]:
                raise RuntimeError("fixed audit cap reached")
            with np.load(path, allow_pickle=False) as arrays:
                t = arrays["t"]
                temperature, oxygen, steam, carbon = (
                    arrays["state__"+key] for key in ("T", "pO2", "pH2O", "carbon"))
                prediction, nuisance, derivative = rate_columns(
                    temperature, oxygen, steam, carbon, parameters, protocol["gas_constant"])
                target_rows, prediction_rows, nuisance_rows, derivative_rows = [], [], [], []
                window = protocol["window_samples"]
                for start in range(0, len(t)-window+1, protocol["stride_samples"]):
                    indices = slice(start, start+window)
                    source_weights, target_weights = weak_weights(t[indices])
                    target_rows.append(np.tensordot(target_weights, carbon[indices], axes=(0, 0)).ravel())
                    prediction_rows.append(np.tensordot(source_weights, prediction[indices], axes=(0, 0)).ravel())
                    nuisance_rows.append(np.tensordot(source_weights, nuisance[indices], axes=(0, 0)).reshape(-1, 2))
                    derivative_rows.append(np.tensordot(source_weights, derivative[indices], axes=(0, 0)).reshape(-1, 5))
                if not target_rows:
                    raise ValueError("saved mission is shorter than the fixed weak window")
                target = np.concatenate(target_rows)
                predicted = np.concatenate(prediction_rows)
                local_nuisance, local_derivative = np.vstack(nuisance_rows), np.vstack(derivative_rows)
                if not all(np.all(np.isfinite(v)) for v in (target, predicted, local_nuisance, local_derivative)):
                    raise ValueError("nonfinite weak source mapping")
                relative_residual = float(np.linalg.norm(target-predicted)/max(np.linalg.norm(target), 1e-14))
                rows.append({
                    "mission": item["mission"], "input_sha256": item["sha256"],
                    "weak_windows": len(target_rows), "weak_rows": len(target),
                    "weak_balance_relative_residual": relative_residual,
                    "saved_model_compatibility_within_fixed_10_percent": relative_residual <= .1,
                    "geometry": diagnostic_record(local_derivative, local_nuisance, profile_information),
                })
                derivatives.append(local_derivative)
                nuisances.append(local_nuisance)
        global_derivative, global_nuisance = np.vstack(derivatives), np.vstack(nuisances)
        independent_nuisance = np.zeros((global_derivative.shape[0], 2*len(rows)))
        offset = 0
        for index, nuisance in enumerate(nuisances):
            independent_nuisance[offset:offset+len(nuisance), 2*index:2*index+2] = nuisance
            offset += len(nuisance)
        global_geometry = diagnostic_record(global_derivative, global_nuisance, profile_information)
        independent_geometry = diagnostic_record(global_derivative, independent_nuisance, profile_information)
        unchanged = all(sha(path) == item["sha256"] for path, item in zip(paths, protocol["inputs"]))
        source_unchanged = all(sha(ROOT / item["path"]) == item["sha256"] for item in identities)
        assessment = {
            "protocol": protocol["version"], "status": "completed", "records": rows,
            "numerical_identity_checks_passed": checks,
            "compatible_saved_model_cases": sum(row["saved_model_compatibility_within_fixed_10_percent"] for row in rows),
            "saved_case_count": len(rows), "global_amplitude_geometry": global_geometry,
            "condition_specific_amplitude_geometry": independent_geometry,
            "saved_carbon_model_compatibility_passed": bool(unchanged and source_unchanged and
                all(row["saved_model_compatibility_within_fixed_10_percent"] for row in rows)),
            "inputs_unchanged": unchanged, "sources_unchanged": source_unchanged,
            "pressure_heat_boundary_or_loading_information_qualified": False,
            "noise_whitening_or_confidence_qualified": False,
            "new_parameter_precision_established": False, "new_estimator_adopted": False,
            "new_fits": 0, "new_trajectories": 0, "new_noise_draws": 0,
            "historical_parameter_results_regraded": False, "original_eight_package_count": "6/8",
            "wall_seconds": time.perf_counter()-started,
        }
        write(output / "assessment.json", assessment)
        write(output / "execution-complete.json", {
            "status": "completed", "assessment_sha256": sha(output / "assessment.json"),
            "input_source_freeze_sha256": sha(output / "input-source-freeze.json"),
            "new_precision_fit_or_confirmation": False,
        })
        print(json.dumps({key: value for key, value in assessment.items()
                          if key not in ("global_amplitude_geometry", "condition_specific_amplitude_geometry")},
                         ensure_ascii=False, indent=2))
    except Exception as error:
        write(output / "failure-retained.json", {"status": "failed_retained", "error": str(error),
             "new_fits": 0, "source_freeze_sha256": sha(output / "input-source-freeze.json")})
        raise


if __name__ == "__main__":
    main()
