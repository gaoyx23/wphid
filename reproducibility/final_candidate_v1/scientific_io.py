"""Fresh observation preparation and saved-input loading for authorized runs."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

from wphid.final_candidate_v1.observations import adapt_condition
from wphid.final_candidate_v1.problem import WeakCondition

PLAN = json.loads((Path(__file__).parent / "execution-plan.json").read_text(encoding="utf-8"))
CONDITIONS = tuple(PLAN["train_conditions"][:5]) + ("hot_oxygen_low", "cool_oxygen_high", "mixed")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT / "artifacts/paper_final_candidate_v1"):
        raise ValueError("Scientific output escaped new-version artifact root")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def save_observations(seed, role, namespace, destination):
    """Generate fresh noise around unchanged manufactured-field references.

    The three role packages in a block share the same field-noise draw, but
    have role-specific compatible source terms. Independent unit = seed block.
    No stored historical observations or completed fits are re-created.
    """
    decisions = json.loads((Path(__file__).parent / "user-decisions.json").read_text(encoding="utf-8"))
    if not decisions["scientific_execution"]["supersedes_earlier_no_new_experiment_instruction"]:
        raise PermissionError("No recorded explicit authorization for new scientific runs")
    destination = Path(destination)
    if not destination.resolve().is_relative_to(ROOT / "artifacts/paper_final_candidate_v1"):
        raise ValueError("Fresh input escaped isolated new-version root")
    if destination.exists():
        raise FileExistsError("Existing observation input must be verified and reused, never regenerated")
    # Reference import is confined to generation, never estimator input loading.
    from experiments import paper_h3_joint_family_topology_partition as reference
    source_lock = json.loads((Path(__file__).parent / "implementation-manifest.json").read_text(encoding="utf-8"))
    for relative, expected in source_lock["frozen_reference_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen generator reference changed: " + relative)
    word = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:4], "little")
    arrays, clipping = {}, {}
    for index, name in enumerate(CONDITIONS):
        t, z, r, clean, forcing = reference._clean(name, role)
        rng = np.random.default_rng(np.random.SeedSequence((seed, index, word)))
        observed = tuple(state+rng.normal(0., .0005*max(float(np.sqrt(np.mean(state*state))), 1e-12), state.shape)
                         for state in clean)
        observed = list(observed)
        clipping[name] = {"oxygen": int(np.count_nonzero(observed[1] < 1e-4)),
                          "damage": int(np.count_nonzero((observed[2] < 1e-8) | (observed[2] > .95)))}
        observed[1] = np.maximum(observed[1], 1e-4)
        observed[2] = np.clip(observed[2], 1e-8, .95)
        condition = adapt_condition(name=name, t=t, z=z, r=r,
                                   observed_states=tuple(observed), supplied_forcing=forcing)
        arrays[name+"_states"] = np.stack(observed)
        arrays[name+"_forcing"] = np.stack(forcing)
        arrays[name+"_bases"] = np.stack(condition.bases)
        arrays[name+"_targets"] = np.stack(condition.targets)
        if index == 0:
            arrays.update({"t": t, "z": z, "r": r,
                           "projection_data": condition.projection.data,
                           "projection_indices": condition.projection.indices,
                           "projection_indptr": condition.projection.indptr,
                           "projection_shape": np.asarray(condition.projection.shape)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    record = {"seed": seed, "role": role, "namespace": namespace,
              "file": destination.relative_to(ROOT).as_posix(), "sha256": sha(destination),
              "conditions": list(CONDITIONS), "grid_shape": [25,25,25],
              "noise_fraction_clean_field_rms": .0005,
              "clipping": clipping, "damage_lower_clip": 1e-8,
              "new_physical_trajectory_integrations": 0,
              "common_field_noise_across_roles_in_same_seed": True,
              "parameters_not_present_in_estimator_input": True}
    save_json(destination.with_suffix(".json"), record)
    return record


def load_conditions(path, expected_sha256):
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError("Saved observation input changed after freezing")
    result = []
    with np.load(path, allow_pickle=False) as data:
        projection = csr_matrix((data["projection_data"], data["projection_indices"], data["projection_indptr"]),
                                shape=tuple(data["projection_shape"]))
        for name in CONDITIONS:
            result.append(WeakCondition(name, tuple(data[name+"_bases"]), tuple(data[name+"_targets"]),
                                        tuple(s.ravel() for s in data[name+"_states"]), projection))
    return tuple(result)
