"""One new sparse derivative/action audit, without repeating matrix-rank checks."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import eye


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/coupled_observation_v1/saved-observation-protocol-20261013.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sparse_bytes(matrix):
    return matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes


def main():
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(1, str(ROOT))
    import wphid
    import wphid.coupled_observation_v1.linearization as module
    from wphid.coupled_balance_v1 import TransportShape, WeakProjector, temporal_tests
    from wphid.coupled_balance_v1.assembler import FIELDS, PRESSURES
    from wphid.coupled_balance_v1.operators import SIDES
    from wphid.coupled_balance_v1.packet_geometry import declared_shell_mesh
    from wphid.coupled_observation_v1 import (
        KineticPoint, ObservationLayout, ObservedBoundary, ObservedValue, SparseWeakMaps, linearize_balances,
    )
    from reproducibility.coupled_observation_v1.qualify_software import verify_dependencies
    if any(not Path(path).resolve().is_relative_to(ROOT / "src") for path in (wphid.__file__, module.__file__)):
        raise SystemExit("main editable import is forbidden")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    verify_dependencies(protocol)
    output = (ROOT / protocol["output_root"]).resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts/paper_coupled_observation_v1"):
        raise SystemExit("independent new output required; completed audits not repeated")
    old = json.loads((ROOT / protocol["input_protocol"]).read_text(encoding="utf-8"))
    if sha(ROOT / protocol["input_protocol"]) != protocol["input_protocol_sha256"]:
        raise SystemExit("locked input protocol changed")
    main_root = Path(protocol["main_root_read_only"]).resolve()
    baseline = json.loads((ROOT / old["baseline_manifest"]).read_text(encoding="utf-8"))
    omitted = {record["path"]: record for record in baseline["excluded"]}
    paths = []
    for item in old["inputs"]:
        path = (main_root / item["path"]).resolve()
        if (not path.is_relative_to(main_root) or item["path"] not in omitted
                or path.stat().st_size != omitted[item["path"]]["bytes"] or sha(path) != item["sha256"]):
            raise SystemExit("locked raw input changed; no regeneration")
        paths.append(path)
    for key in ("case_manifest", "saved_model"):
        if sha(ROOT / old[key]["path"]) != old[key]["sha256"]:
            raise SystemExit("declared geometry or saved point changed")
    geometry = json.loads((ROOT / old["case_manifest"]["path"]).read_text(encoding="utf-8"))["case"]["geometry"]
    saved = json.loads((ROOT / old["saved_model"]["path"]).read_text(encoding="utf-8"))["results"]["chemistry"]
    if not saved["success"]:
        raise SystemExit("saved parameter point unavailable")
    parameters = saved["parameters"]
    point = KineticPoint(parameters["oxygen_activation_energy"], parameters["steam_activation_energy"],
        parameters["oxygen_exponent"], parameters["steam_exponent"], parameters["carbon_exponent"])
    relative = parameters["steam_pre_exponential"]/parameters["oxygen_pre_exponential"]
    identities = [{"path": path, "sha256": sha(ROOT / path)} for path in protocol["source_files"]]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "input-source-freeze.json", {
        "protocol": protocol["version"], "cwd": str(ROOT), "imported_wphid": str(Path(wphid.__file__).resolve()),
        "imported_observation_module": str(Path(module.__file__).resolve()), "output_root": str(output),
        "inputs": old["inputs"], "source_files": identities, "candidate": old["candidate"],
        "kinetics_from_saved_fit": parameters, "probe": protocol["action_probe"],
        "no_material_or_loading_truth_used": True,
    })
    try:
        records = []
        for item, path in zip(old["inputs"], paths):
            case_started = time.perf_counter()
            if case_started-started > protocol["wall_cap_seconds"]:
                raise RuntimeError("fixed derivative audit cap reached")
            with np.load(path, allow_pickle=False) as arrays:
                state_arrays = {key: arrays["state__"+key] for key in FIELDS}
                nt, ns, nx = state_arrays["T"].shape
                mesh, _ = declared_shell_mesh(geometry, ns, nx)
                tests = temporal_tests(arrays["t"], window_samples=old["window_samples"], stride_samples=old["stride_samples"])
                projector = WeakProjector(mesh, tests, np.eye(ns*nx).reshape(ns*nx, ns, nx))
                load_names = ("internal_oxygen_pressure", "external_steam_pressure", "external_oxygen_pressure", "surface_heat_flux")
                layout = ObservationLayout([(key, (nt, ns, nx)) for key in FIELDS]+
                    [("load."+key, (nt, ns)) for key in load_names])
                states = {key: layout.observed(key, value) for key, value in state_arrays.items()}
                loads = {key: layout.observed("load."+key, arrays["load__"+key]) for key in load_names}
                external = {("pO2", "inner"): loads["internal_oxygen_pressure"],
                            ("pH2O", "inner"): loads["external_steam_pressure"],
                            ("pProducts", "inner"): ObservedValue.fixed(np.zeros((nt, ns)), layout.size),
                            ("pO2", "outer"): loads["external_oxygen_pressure"]}
                barriers = {key: ObservedValue.fixed(np.ones((nt, ns)), layout.size) for key in external}
                sealed = frozenset((key, side) for key in PRESSURES for side in SIDES if (key, side) not in external)
                boundary = ObservedBoundary(external, barriers, sealed, {"inner": loads["surface_heat_flux"]},
                    {"outer": ObservedValue.fixed(np.full((nt, ns), old["candidate"]["ambient_temperature_assumption_K"]), layout.size)},
                    frozenset(("inlet", "outlet")))
                maps = SparseWeakMaps(projector)
                balance, linearization, domain = linearize_balances(projector, layout, states, kinetics=point,
                    relative_rate=relative, transport=TransportShape(1., 1., 1., 0.),
                    gas_modulation=ObservedValue.fixed(np.ones((nt, ns, nx)), layout.size),
                    thermal_modulation=ObservedValue.fixed(np.ones((nt, ns, nx)), layout.size),
                    thermal_direction_ratio=1., heat_knots=tuple(old["candidate"]["heat_capacity_knots_K"]),
                    boundary=boundary, maps=maps)
                matrices = (linearization.target_jacobian, *linearization.column_jacobians)
                coefficients = np.asarray(protocol["action_probe"]["linear_coefficients"], dtype=float)
                # Unit factor is an algebra-only probe, not calibrated physical noise.
                noise = linearization.noise_operator(coefficients, eye(layout.size, format="csr"))
                vector = np.cos(np.arange(noise.shape[0])/7.)
                action, diagonal = noise.matvec(vector), noise.diagonal()
                quadratic = float(vector @ action)
                pulled = np.asarray(noise.root.T @ vector).ravel()
                exact_quadratic = float(pulled @ pulled)
                energy_defect = abs(quadratic-exact_quadratic)/max(abs(exact_quadratic), 1.)
                action_valid = bool(np.all(np.isfinite(action)) and np.all(np.isfinite(diagonal))
                                    and np.all(diagonal >= 0) and energy_defect <= protocol["action_identity_rtol"])
                if not action_valid:
                    raise ValueError("sparse covariance action/energy identity failed")
                records.append({
                    "mission": item["mission"], "input_sha256": item["sha256"],
                    "observation_derivative_shape": list(linearization.target_jacobian.shape),
                    "state_coordinate_count": 5*nt*ns*nx, "boundary_load_coordinate_count": 4*nt*ns,
                    "raw_coordinate_count": layout.size, "target_nnz": int(matrices[0].nnz),
                    "column_nnz": {key: int(matrix.nnz) for key, matrix in zip(balance.column_keys, matrices[1:])},
                    "total_derivative_nnz": int(sum(matrix.nnz for matrix in matrices)),
                    "csr_derivative_bytes": int(sum(sparse_bytes(matrix) for matrix in matrices)),
                    "dense_equivalent_bytes_not_allocated": int(len(matrices)*matrices[0].shape[0]*layout.size*8),
                    "source_domain_report": domain,
                    "all_observation_derivatives_finite": all(bool(np.all(np.isfinite(matrix.data))) for matrix in matrices),
                    "sparse_covariance_action_valid": action_valid, "action_energy_relative_defect": energy_defect,
                    "dense_global_jacobian_or_covariance_allocated": False,
                    "matrix_rank_audit_repeated": False, "parameter_fit_or_noise_draws": 0,
                    "candidate_remains_unmodulated_declared_point": True,
                    "physical_noise_calibration_or_observation_law_qualified": False,
                    "case_wall_seconds": time.perf_counter()-case_started,
                })
                del matrices, noise, linearization, maps, states, loads, boundary, balance, projector
        unchanged_inputs = all(sha(path) == item["sha256"] for item, path in zip(old["inputs"], paths))
        unchanged_sources = all(sha(ROOT / item["path"]) == item["sha256"] for item in identities)
        assessment = {
            "protocol": protocol["version"], "status": "completed", "records": records,
            "saved_case_count": len(records), "sparse_derivative_and_action_interface_passed": bool(
                unchanged_inputs and unchanged_sources and all(record["all_observation_derivatives_finite"] and
                    record["sparse_covariance_action_valid"] for record in records)),
            "inputs_unchanged": unchanged_inputs, "sources_unchanged": unchanged_sources,
            "new_fits": 0, "new_trajectories": 0, "new_noise_draws": 0,
            "actual_noise_or_clipping_law_qualified": False, "scientific_parameter_precision_established": False,
            "new_estimator_adopted": False, "original_eight_package_count": "6/8", "wall_seconds": time.perf_counter()-started,
        }
        write(output / "assessment.json", assessment)
        write(output / "execution-complete.json", {
            "status": "completed", "assessment_sha256": sha(output / "assessment.json"),
            "input_source_freeze_sha256": sha(output / "input-source-freeze.json"),
            "sparse_derivative_and_action_interface_passed": assessment["sparse_derivative_and_action_interface_passed"],
            "scientific_precision_or_confirmation": False,
        })
        print(json.dumps({key: value for key, value in assessment.items() if key != "records"}, ensure_ascii=False, indent=2))
        if not assessment["sparse_derivative_and_action_interface_passed"]:
            raise SystemExit(1)
    except Exception as error:
        write(output / "failure-retained.json", {"status": "failed_retained", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
