from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import numpy as np

spec=spec_from_file_location("paired_holdout",Path(__file__).resolve().parents[2]/"experiments/confirm_fixed_paired_noise_holdout.py")
module=module_from_spec(spec);spec.loader.exec_module(module)


def test_new_seed_and_budget_are_frozen_and_distinct():
    assert module.SEED not in (6100819,2026100803)
    assert module.REPEATS==128
    assert module.METHOD=="first_order_zero_mean"


def test_noise_packet_reproducibility_and_fixed_drivers():
    clean={n:np.ones((3,3,3))*(500. if n=="T" else .2 if n=="crack" else 1.) for n in module.FIELDS}
    first=module.draw_observations(clean,0,0);second=module.draw_observations(clean,0,0)
    for n in clean:np.testing.assert_array_equal(first[n],second[n])
    np.testing.assert_array_equal(first["carbon"],clean["carbon"])
    np.testing.assert_array_equal(first["stress"],clean["stress"])


def test_summary_keeps_empty_outputs_separate_from_wrong_uniques():
    observations=[dict(decision=dict(candidate_set=c)) for c in (["a"],[],["b"],["a","b"])]
    summary=module.summarize("a",observations)
    assert summary["truth_retained"]==2
    assert summary["unique_correct"]==1
    assert summary["wrong_unique"]==1
    assert summary["empty"]==1
    assert not summary["frozen_noise_holdout_target_met"]
