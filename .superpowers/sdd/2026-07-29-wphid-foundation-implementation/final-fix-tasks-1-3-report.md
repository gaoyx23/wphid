# Final-fix Tasks 1-3 checkpoint report

Date: 2026-07-30
Scope: committed final-fix Tasks 1-3 only
Checkpoint commit: the enclosing commit with subject `fix: checkpoint final-fix tasks 1-3`

## Status

The focused Task 1-3 suite is green. The current genuine 5% multivariate E1 gate is **NO-GO** because the minimum per-seed support F1 is below the frozen threshold. The one requested full-suite run stops during collection in the untouched E3 code; no E3, dependency, README, gate-report, artifact, or benchmark-expansion work was performed.

## Before and after tests

Focused command:

```text
python -m pytest tests/test_types.py tests/grammar/test_generate.py tests/integration/test_e1_closed.py -q
```

Before the final checkpoint fixes:

```text
9 failed, 27 passed in 6.04s
```

The nine failures were seven legacy calls to the now-parameterized `closed_candidate_library`, one assertion against obsolete top-level `seed` metadata, and one assertion against the retired three-condition noise set.

After the final checkpoint fixes:

```text
36 passed in 6.01s
```

Full-suite command, run exactly once after the focused suite became green:

```text
python -m pytest -q
```

Observed result:

```text
ERROR tests/integration/test_e3_missing.py
ImportError: cannot import name 'ARRHENIUS_TEMPERATURE' from 'experiments.e1_closed_library'
Interrupted: 1 error during collection in 1.95s
```

This is an out-of-scope E3 compatibility failure. Because collection stopped at that point, additional full-suite failures may remain latent. They were not investigated or fixed.

## Implemented Task 1-3 behavior

- Equation-level dimensional contracts now require each candidate's raw feature dimension plus declared coefficient dimension to equal the derivative target dimension before weak projection.
- Candidate canonical identities preserve nonlinear parameter values with round-trip-safe representations.
- `DiscoveryResult` validates names, coefficients, support, error, and diagnostics while retaining immutable copies.
- The closed E1 library requires an explicit Arrhenius temperature; legacy Task 1 tests pass an explicit test profile rather than restoring a hidden module-level truth parameter.
- E1 discovery receives a declared theta grid and selects theta by leave-one-training-condition-out profile scoring. Validation observations remain excluded from theta selection; the validation-perturbation regression test passes.
- Truth theta, support, and coefficients are read only in post-discovery evaluation metrics.
- E1 applies independent Gaussian noise at 5% of each clean field's per-condition RMS to `T`, `pO2`, `sigma`, `Df`, and `Do` for all six frozen E1 conditions.
- The non-random run label is persisted as `provenance_id`; actual replicate randomness remains represented by the five per-seed records.

## Files in the checkpoint

Created:

- `src/wphid/dimensions.py`
- `.superpowers/sdd/2026-07-29-wphid-foundation-implementation/final-fix-tasks-1-3-report.md`

Modified:

- `experiments/e1_closed_library.py`
- `src/wphid/grammar/atoms.py`
- `src/wphid/grammar/generate.py`
- `src/wphid/pipeline.py`
- `src/wphid/truth/aging_ode.py`
- `src/wphid/types.py`
- `tests/grammar/test_generate.py`
- `tests/integration/test_e1_closed.py`
- `tests/test_types.py`

No generated artifacts are included.

## Frozen E1 configuration

- Provenance ID: `20260729`
- Noise seeds: `101`, `203`, `307`, `401`, `509`
- Noise: independent zero-mean Gaussian, `0.05 * RMS(clean field)` per condition and field
- Observed fields: `T`, `pO2`, `sigma`, `Df`, `Do`
- Training conditions: `thermal_hot`, `fatigue_warm`, `fatigue_cool`, `fatigue_low_oxygen`
- Validation conditions: `validation_hot_fatigue`, `validation_cool_fatigue`
- Theta grid: `1200, 1400, 1600, 1800, 2000, 2200, 2400`
- Theta selection data: training conditions only
- Frozen gate: minimum per-seed support F1 >= `0.90`

## Current 5% E1 metrics

| Seed | Precision | Recall | F1 | False positives | Selected theta | Theta rel. error | Coefficient rel. L2 | Validation NRMSE | Seed gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 101 | 1.00 | 1.00 | 1.000000 | 0 | 1800 | 0.000000 | 0.129577 | 0.112176 | PASS |
| 203 | 0.60 | 1.00 | 0.750000 | 2 | 2400 | 0.333333 | 2.757555 | 0.101813 | FAIL |
| 307 | 1.00 | 1.00 | 1.000000 | 0 | 1800 | 0.000000 | 0.269230 | 0.105590 | PASS |
| 401 | 0.75 | 1.00 | 0.857143 | 1 | 2400 | 0.333333 | 3.697704 | 0.147074 | FAIL |
| 509 | 0.60 | 1.00 | 0.750000 | 2 | 2400 | 0.333333 | 5.057196 | 0.144591 | FAIL |

Aggregate values:

- Passing seeds: `2/5`
- Minimum support F1: `0.750000`
- Mean support precision: `0.790000`
- Mean support recall: `1.000000`
- Maximum coefficient relative L2 error: `5.057196`
- Maximum theta relative error: `0.333333`
- Maximum validation NRMSE: `0.147074`

**E1 gate result: NO-GO.** The observed minimum F1 is `0.75`, below the frozen `0.90` threshold. Failing seeds are `203`, `401`, and `509`. This result is not overridden by the green focused contract tests.

## Scratch and working-tree hygiene

`.codex-tmp` contained only two untracked intermediate copies of `experiments/e1_closed_library.py` under `a/` and `b/`. It was verified as agent scratch and removed. No repository artifact was generated or staged.

## Concerns and deferred work

- E3 still imports the removed fixed E1 `ARRHENIUS_TEMPERATURE` constant. Migrating that E3 caller is intentionally deferred because this checkpoint must not start E3.
- The full suite stopped at E3 collection, so its result is not evidence that all non-focused tests pass.
- The E1 frozen gate is a real scientific NO-GO at 5% multivariate noise despite perfect recall; spurious pressure terms and theta selection at 2400 degrade seeds 203, 401, and 509.
- E1 has the train/validation roles defined by the committed final-fix design; no independent E1 test split was invented in this checkpoint.
- E2 remains unimplemented, and this checkpoint supports no full-WPHID design claim.
