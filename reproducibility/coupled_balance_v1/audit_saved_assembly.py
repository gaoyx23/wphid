"""One new matrix-assembly diagnostic on locked saved arrays; no refitting."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/coupled_balance_v1/saved-assembly-protocol-20261013.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the algorithm worktree")
    sys.path.insert(0, str(ROOT / "src"))
    import wphid
    import wphid.coupled_balance_v1.assembler as module
    from wphid.coupled_balance_v1 import (BoundaryInputs, Modulation, TransportShape,
        WeakProjector, arrhenius_pair, assemble_balances, temporal_tests)
    from wphid.coupled_balance_v1.assembler import FIELDS, PRESSURES
    from wphid.coupled_balance_v1.operators import SIDES
    from wphid.coupled_balance_v1.packet_geometry import declared_shell_mesh
    for path in (Path(wphid.__file__).resolve(), Path(module.__file__).resolve()):
        if not path.is_relative_to(ROOT / "src"):
            raise SystemExit("unexpected main editable install")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    output = (ROOT / protocol["output_root"]).resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts/paper_coupled_balance_v1"):
        raise SystemExit("new independent output required; finished audits are not repeated")
    previous = ROOT / protocol["prior_array_completion"]
    complete = json.loads(previous.read_text(encoding="utf-8"))
    if (not complete["raw_array_qualification_passed"] or
            sha(previous.parent / "assessment.json") != complete["assessment_sha256"]):
        raise SystemExit("saved array qualification incomplete or changed")
    main_root = Path(protocol["main_root_read_only"]).resolve()
    baseline = json.loads((ROOT / protocol["baseline_manifest"]).read_text(encoding="utf-8"))
    omitted = {item["path"]: item for item in baseline["excluded"]}
    for key in ("case_manifest", "saved_model"):
        if sha(ROOT / protocol[key]["path"]) != protocol[key]["sha256"]:
            raise SystemExit("declared geometry/model identity changed")
    geometry = json.loads((ROOT / protocol["case_manifest"]["path"]).read_text(encoding="utf-8"))["case"]["geometry"]
    # Geometry only: material/chemistry truth stored elsewhere is not used.
    saved = json.loads((ROOT / protocol["saved_model"]["path"]).read_text(encoding="utf-8"))["results"]["chemistry"]
    if not saved["success"]:
        raise SystemExit("selected saved model is not successful")
    kinetics = saved["parameters"]
    relative_rate = kinetics["steam_pre_exponential"]/kinetics["oxygen_pre_exponential"]
    paths = []
    for item in protocol["inputs"]:
        path = (main_root / item["path"]).resolve()
        if (not path.is_relative_to(main_root) or item["path"] not in omitted
                or path.stat().st_size != omitted[item["path"]]["bytes"] or sha(path) != item["sha256"]):
            raise SystemExit("saved input hash/baseline mismatch; no regeneration")
        paths.append(path)
    for root in protocol["required_balance_qualifications"]:
        folder = ROOT / root
        receipt = json.loads((folder / "execution-complete.json").read_text(encoding="utf-8"))
        if not receipt["software_passed"] or sha(folder / "assessment.json") != receipt["assessment_sha256"]:
            raise SystemExit("required software qualification missing or changed")
        freeze = json.loads((folder / "source-freeze.json").read_text(encoding="utf-8"))
        for record in freeze["sources"]:
            if sha(ROOT / record["path"]) != record["sha256"]:
                raise SystemExit("qualified frozen balance dependency changed")
    identities = [{"path": p, "sha256": sha(ROOT / p)} for p in protocol["source_files"]]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "input-source-freeze.json", {
        "protocol": protocol["version"], "cwd": str(ROOT), "imported_wphid": str(Path(wphid.__file__).resolve()),
        "imported_assembler": str(Path(module.__file__).resolve()), "output_root": str(output),
        "inputs": protocol["inputs"], "source_files": identities, "geometry_used": geometry,
        "candidate": protocol["candidate"], "kinetics_from_saved_pooled_fit": kinetics,
        "relative_rate_from_saved_fit": relative_rate, "material_or_loading_truth_used": False,
    })
    try:
        records = []
        for item, path in zip(protocol["inputs"], paths):
            if time.perf_counter()-started > protocol["wall_cap_seconds"]:
                raise RuntimeError("fixed assembly audit cap reached")
            with np.load(path, allow_pickle=False) as arrays:
                states = {key: arrays["state__"+key] for key in FIELDS}
                nt, ns, nx = states["T"].shape
                mesh, coordinates = declared_shell_mesh(geometry, ns, nx)
                errors = {key: float(np.max(np.abs(arrays[key]-value))) for key, value in coordinates.items()}
                if any(error > protocol["coordinate_absolute_tolerance"] for error in errors.values()):
                    raise ValueError("saved shell coordinates do not match the declared geometry")
                tests = temporal_tests(arrays["t"], window_samples=protocol["window_samples"],
                                       stride_samples=protocol["stride_samples"])
                spatial = np.eye(ns*nx).reshape(ns*nx, ns, nx)
                projector = WeakProjector(mesh, tests, spatial)
                rates = arrhenius_pair(states, oxygen_energy=kinetics["oxygen_activation_energy"],
                    steam_energy=kinetics["steam_activation_energy"], oxygen_order=kinetics["oxygen_exponent"],
                    steam_order=kinetics["steam_exponent"], carbon_order=kinetics["carbon_exponent"])
                external = {("pO2", "inner"): arrays["load__internal_oxygen_pressure"],
                            ("pH2O", "inner"): arrays["load__external_steam_pressure"],
                            ("pProducts", "inner"): np.zeros((nt, ns)),
                            ("pO2", "outer"): arrays["load__external_oxygen_pressure"]}
                barriers = {key: Modulation(np.ones((nt, ns))) for key in external}
                sealed = frozenset((key, side) for key in PRESSURES for side in SIDES if (key, side) not in external)
                boundary = BoundaryInputs(external, barriers, sealed,
                    {"inner": arrays["load__surface_heat_flux"]},
                    {"outer": np.full((nt, ns), protocol["candidate"]["ambient_temperature_assumption_K"])},
                    frozenset(("inlet", "outlet")))
                system = assemble_balances(projector, states, rates, relative_rate=relative_rate,
                    transport=TransportShape(1., 1., 1., 0.), gas_modulation=Modulation(np.ones((nt, ns, nx))),
                    thermal_modulation=Modulation(np.ones((nt, ns, nx))), thermal_direction_ratio=1.,
                    heat_knots=tuple(protocol["candidate"]["heat_capacity_knots_K"]), boundary=boundary)
                column_norm = np.linalg.norm(system.matrix, axis=0)
                active = column_norm > 0
                normalized = system.matrix[:, active]/column_norm[active]
                spectrum = np.linalg.svd(normalized, compute_uv=False)
                rank = int(np.count_nonzero(spectrum > protocol["numerical_rank_rtol"]*spectrum[0])) if len(spectrum) else 0
                finite = (np.all(np.isfinite(system.matrix)) and np.all(np.isfinite(system.target))
                          and all(np.all(np.isfinite(value)) for value in system.matrix_jacobian.values()))
                zeros = {key: int(np.count_nonzero(states[key] <= 0)) for key in ("pO2", "pH2O")}
                records.append({
                    "mission": item["mission"], "input_sha256": item["sha256"],
                    "state_shape": [nt, ns, nx], "coordinate_max_absolute_errors": errors,
                    "matrix_shape": list(system.matrix.shape), "column_keys": list(system.column_keys),
                    "all_targets_columns_and_derivatives_finite": bool(finite),
                    "outer_derivative_keys": list(system.matrix_jacobian),
                    "linear_column_span_rank": rank, "linear_column_count": len(system.column_keys),
                    "column_normalized_singular_values": spectrum.tolist(),
                    "column_normalization_purpose": "numerical span only, not observation weighting",
                    "nonpositive_raw_pressure_nodes": zeros,
                    "raw_state_coordinate_count": 5*nt*ns*nx,
                    "practical_native_observation_jacobian_qualified": False,
                    "engineering_model_compatibility_tested": False,
                    "new_parameter_precision_established": False,
                })
        unchanged_inputs = all(sha(path) == item["sha256"] for path, item in zip(paths, protocol["inputs"]))
        unchanged_sources = all(sha(ROOT / item["path"]) == item["sha256"] for item in identities)
        assessment = {
            "protocol": protocol["version"], "status": "completed", "records": records,
            "assembled_saved_case_count": len(records), "saved_array_count": len(paths),
            "assembly_interface_passed": bool(unchanged_inputs and unchanged_sources and
                all(record["all_targets_columns_and_derivatives_finite"] for record in records)),
            "inputs_unchanged": unchanged_inputs, "sources_unchanged": unchanged_sources,
            "candidate_is_unmodulated_declared_point": True,
            "source_material_or_loading_truth_used": False,
            "no_scientific_parameter_or_nuisance_fit": True,
            "native_noise_calibration_or_clipping_qualified": False,
            "coupling_information_after_complete_unknowns_established": False,
            "new_estimator_adopted": False, "new_fits": 0, "new_trajectories": 0, "new_noise_draws": 0,
            "original_eight_package_count": "6/8", "wall_seconds": time.perf_counter()-started,
        }
        write(output / "assessment.json", assessment)
        write(output / "execution-complete.json", {
            "status": "completed", "assessment_sha256": sha(output / "assessment.json"),
            "input_source_freeze_sha256": sha(output / "input-source-freeze.json"),
            "assembly_interface_passed": assessment["assembly_interface_passed"],
            "scientific_precision_or_confirmation": False,
        })
        print(json.dumps({key: value for key, value in assessment.items() if key != "records"},
                         ensure_ascii=False, indent=2))
    except Exception as error:
        write(output / "failure-retained.json", {"status": "failed_retained", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
