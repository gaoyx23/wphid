"""Read saved arrays only: no simulation, loader fallback, fitting, or scoring."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/source_geometry_v6/saved-input-audit-protocol.json"


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
    package = Path(wphid.__file__).resolve()
    if not package.is_relative_to(ROOT / "src"):
        raise SystemExit("unexpected main editable install")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    output = (ROOT / protocol["output_root"]).resolve()
    if not output.is_relative_to(ROOT / "artifacts/paper_source_geometry_v6") or output.exists():
        raise SystemExit("independent output must be new and inside the local artifact root")
    main_root = Path(protocol["main_root_read_only"]).resolve()
    baseline = json.loads((ROOT / protocol["baseline_manifest"]).read_text(encoding="utf-8"))
    omitted = {entry["path"]: entry for entry in baseline["excluded"]}
    qualification = json.loads((ROOT / protocol["historical_result"]).read_text(encoding="utf-8"))
    case_manifest = json.loads((ROOT / protocol["case_manifest"]).read_text(encoding="utf-8"))
    source_identities = [{"path": p, "sha256": sha(ROOT / p)} for p in protocol["source_files"]]
    paths = []
    for item in protocol["inputs"]:
        path = (main_root / item["path"]).resolve()
        if not path.is_relative_to(main_root) or item["path"] not in omitted:
            raise SystemExit("input is not a registered read-only main artifact")
        if path.stat().st_size != omitted[item["path"]]["bytes"]:
            raise SystemExit("baseline input byte identity mismatch")
        if sha(path) != item["sha256"]:
            raise SystemExit("input hash changed since protocol freeze")
        if item["mission"] not in case_manifest["missions"]:
            raise SystemExit("mission is not registered in the saved case manifest")
        paths.append(path)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "input-source-freeze.json", {
        "protocol": protocol["version"], "cwd": str(ROOT), "imported_wphid": str(package),
        "output_root": str(output), "main_root_read_only": str(main_root),
        "inputs": protocol["inputs"], "sources": source_identities,
        "array_copy_or_write": False, "generator_called": False,
    })
    records = []
    for item, path in zip(protocol["inputs"], paths):
        if time.perf_counter() - started > protocol["wall_cap_seconds"]:
            raise SystemExit("input audit cap reached; partial attempt retained")
        with np.load(path, allow_pickle=False) as saved:
            states = {key.removeprefix("state__"): saved[key] for key in saved.files
                      if key.startswith("state__")}
            required = tuple(protocol["required_fields"])
            if any(name not in states for name in required):
                raise SystemExit("required observed field missing; no regeneration")
            t, s, x = (saved[name] for name in ("t", "s", "x"))
            expected_shape = (len(t), len(s), len(x))
            finite = all(np.all(np.isfinite(states[name])) for name in required)
            same_shape = all(states[name].shape == expected_shape for name in required)
            valid_grid = all(np.all(np.isfinite(v)) and np.all(np.diff(v) > 0)
                             for v in (t, s, x))
            field_stats = {}
            for name in required:
                values = states[name]
                field_stats[name] = {
                    "shape": list(values.shape), "min": float(np.min(values)),
                    "max": float(np.max(values)), "range": float(np.ptp(values)),
                    "zero_count": int(np.count_nonzero(values == 0)),
                    "zero_count_after_initial": int(np.count_nonzero(values[1:] == 0)),
                }
            carbon = states["carbon"]
            native_carbon_floor_hit = bool(np.any(carbon <= .02))
            carbon_monotone = bool(np.max(np.diff(carbon, axis=0)) <= 1e-12)
            historical = qualification["qualification"]["reference_bartz"]["conditions"][item["mission"]]
            pairs = {
                "peak_temperature_k": field_stats["T"]["max"],
                "minimum_carbon_fraction": field_stats["carbon"]["min"],
                "maximum_crack_damage": field_stats["crack"]["max"],
                "oxygen_pressure_range_pa": field_stats["pO2"]["range"],
                "steam_pressure_range_pa": field_stats["pH2O"]["range"],
                "product_pressure_range_pa": field_stats["pProducts"]["range"],
            }
            matches = {key: bool(np.isclose(value, historical[key], rtol=1e-12, atol=1e-12))
                       for key, value in pairs.items()}
            records.append({
                "mission": item["mission"], "input_sha256": item["sha256"],
                "read_location": "main_read_only", "keys": list(saved.files),
                "time_range_seconds": [float(t[0]), float(t[-1])],
                "output_time_steps_seconds": np.unique(np.round(np.diff(t), 10)).tolist(),
                "all_required_arrays_finite": finite, "all_field_shapes_match": same_shape,
                "grid_finite_and_increasing": valid_grid, "fields": field_stats,
                "carbon_sampled_floor_hit": native_carbon_floor_hit,
                "carbon_sampled_nonincreasing": carbon_monotone,
                "historical_raw_extrema_match": matches,
                "arrays_usable_for_fullfield_synthetic_development": bool(
                    finite and same_shape and valid_grid and all(matches.values())
                    and not native_carbon_floor_hit and carbon_monotone),
                "pressure_clipping_excluded": False,
            })
    unchanged_sources = all(sha(ROOT / record["path"]) == record["sha256"]
                            for record in source_identities)
    unchanged_inputs = all(sha(path) == item["sha256"] for item, path in zip(protocol["inputs"], paths))
    assessment = {
        "protocol": protocol["version"], "status": "completed", "records": records,
        "saved_array_count": len(records),
        "usable_saved_arrays": sum(r["arrays_usable_for_fullfield_synthetic_development"] for r in records),
        "input_identities_unchanged": unchanged_inputs,
        "source_identities_unchanged": unchanged_sources,
        "raw_array_qualification_passed": bool(unchanged_inputs and unchanged_sources and
            all(r["arrays_usable_for_fullfield_synthetic_development"] for r in records)),
        "independent_instrument_calibration_established": False,
        "exact_physical_stoichiometry_established": False,
        "pressure_clipping_excluded": False,
        "full_source_balance_and_boundary_qualification_established": False,
        "historical_generator_source_hash_recorded_in_case_manifest": False,
        "new_scientific_development_gate_passed": False,
        "new_confirmation_allowed": False,
        "new_fits": 0, "new_trajectories": 0, "new_noise_draws": 0,
        "historical_parameter_results_regraded": False,
        "wall_seconds": time.perf_counter() - started,
    }
    write(output / "assessment.json", assessment)
    write(output / "execution-complete.json", {
        "status": "completed", "assessment_sha256": sha(output / "assessment.json"),
        "input_source_freeze_sha256": sha(output / "input-source-freeze.json"),
        "raw_array_qualification_passed": assessment["raw_array_qualification_passed"],
        "scientific_performance_run": False,
    })
    print(json.dumps({key: value for key, value in assessment.items() if key != "records"},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if assessment["raw_array_qualification_passed"] else 1)


if __name__ == "__main__":
    main()
