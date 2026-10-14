"""Authorized cost-development packet; no confirmation accuracy inspected."""

import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(Path(__file__).parent))

from scientific_io import save_observations, load_conditions, save_json, PLAN
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import Candidate, Mechanism, FitBudget, fit_candidate
from wphid.final_candidate_v1.adam_adapter import AdamBudget, fit_adam
from wphid.final_candidate_v1.pipeline import prepare_views


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Wrong worktree")
    output = ROOT / "artifacts/paper_final_candidate_v1/cost_development_20261011"
    path = output / "independent_9101091.npz"
    if path.exists():
        raise FileExistsError("Cost development already prepared; read its saved report instead of rerunning")
    start = time.perf_counter()
    record = save_observations(9101091, "independent", "WPHID_COST_DEVELOPMENT_20261011_V1", path)
    conditions = load_conditions(path, record["sha256"])
    family = next(f for f in paper_families() if f.name == "arrhenius_oxygen_damage")
    candidate = Candidate((Mechanism("reaction", family, (0,1,2)),))
    training, _ = prepare_views(conditions, (family,), tuple(PLAN["train_conditions"]), tuple(PLAN["heldout_conditions"]))
    wbudget = FitBudget(seeds=(11011,), initial_fractions=(.5,), global_iterations=1,
                        local_iterations=5, maximum_objective_calls=500, maximum_cpu_seconds=60.)
    abudget = AdamBudget(starts=FitBudget(seeds=(42,), initial_fractions=(.5,), maximum_cpu_seconds=60.),
                        epochs_per_start=100, maximum_backward_calls=100)
    wfit = fit_candidate(training, candidate, wbudget)
    afit = fit_adam(training, candidate, abudget)
    cost = {"scope": "independent-role cost development only, no truth-based accuracy or confirmation selection",
            "input": record, "input_preparation_wall_seconds": time.perf_counter()-start-wfit.wall_seconds-afit["fit"].wall_seconds,
            "wphid": {"cpu_seconds": wfit.cpu_seconds, "objective_calls": wfit.objective_calls},
            "adam": {"cpu_seconds": afit["fit"].cpu_seconds, "wall_seconds": afit["fit"].wall_seconds,
                      "backward_calls": afit["backward_calls"], "forward_calls": afit["forward_calls"]},
            "truth_errors_calculated": False, "confirmation_inputs_generated": 0,
            "research_program_estimates_in_cost_development": 2,
            "matched_Q1_estimates_completed": 0,
            "common_cap_proposal_cpu_seconds": 600.,
            "cost_cap_establishes_convergence": False}
    save_json(output / "cost.json", cost)
    import json
    print(json.dumps(cost, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
