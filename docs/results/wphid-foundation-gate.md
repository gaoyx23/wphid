# WPHID foundation discovery gate

## Decision

**NO-GO: temporal foundation gate not passed.**

This decision is limited to the declared temporal weak-form benchmark and its finite physics grammar. It does not claim general equation discovery or broad noise robustness.

## Reproduction

From the repository root:

```text
python -m pip install -e ".[test]"
python -m pytest -q
python experiments/e1_closed_library.py
python experiments/e3_missing_interaction.py
python experiments/foundation_report.py
```

## E1: closed-library recovery

- Noise: 5% independent zero-mean Gaussian per field and time sample
- Applied to: T, pO2, sigma, Df, Do
- Theta grid: 1200--2400 (7 points)
- Train conditions: thermal_hot, fatigue_warm, fatigue_cool, fatigue_low_oxygen
- Validation conditions: validation_hot_fatigue, validation_cool_fatigue
- Theta selection: training_conditions_only
- Gate: minimum_per_seed_support_f1 greater_than_or_equal 0.9
- Observed: 0.75
- Passing seeds: 2/5
- Mean support precision: 0.79
- Mean support recall: 1
- Max coefficient relative L2 error: 5.0572
- Max theta relative error: 0.333333
- Reference validation NRMSE: 0.147074

| Seed | Precision | Recall | F1 | False positives | Selected theta | Theta rel. error | Coeff rel. L2 | Validation NRMSE | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 101 | 1 | 1 | 1 | 0 | 1800 | 0 | 0.129577 | 0.112176 | PASS |
| 203 | 0.6 | 1 | 0.75 | 2 | 2400 | 0.333333 | 2.75756 | 0.101813 | FAIL |
| 307 | 1 | 1 | 1 | 0 | 1800 | 0 | 0.26923 | 0.10559 | PASS |
| 401 | 0.75 | 1 | 0.857143 | 1 | 2400 | 0.333333 | 3.6977 | 0.147074 | FAIL |
| 509 | 0.6 | 1 | 0.75 | 2 | 2400 | 0.333333 | 5.0572 | 0.144591 | FAIL |

## E3: missing-interaction discovery

- Noise: 0.0001 (target-only Do observation noise; t, T, pO2, sigma, and Df are unchanged)
- Expected missing interaction: arr(T;theta=1800)*pO2*Df
- Detection rate: 0.6 (3/5)
- Max false interactions: 0
- Max expanded test NRMSE / E1 reference: 0.514227
- E1 reference validation NRMSE: 0.147074

| Seed | Detected | Admitted | False interactions | Test NRMSE / E1 ref | Gate |
|---:|:---:|---|:---:|---:|:---:|
| 11 | yes | arr(T;theta=1800)*pO2*Df | 0 | 0.00151526 | PASS |
| 23 | yes | arr(T;theta=1800)*pO2*Df | 0 | 0.00113775 | PASS |
| 47 | yes | arr(T;theta=1800)*pO2*Df | 0 | 0.00165278 | PASS |
| 89 | no | none | 0 | 0.514227 | FAIL |
| 131 | no | none | 0 | 0.2854 | FAIL |

## Scope and limitations

- E2 (redundant library) is unimplemented and not evaluated.
- This benchmark is temporal only. Spatial discretization, boundary handling, field noise, covariate noise, and out-of-grammar failure modes are reserved for Plan 2.
- The finite physics grammar is declared a priori; this is not unrestricted symbolic discovery.
- The overall gate is NO-GO. E1 minimum F1 = 0.75 (threshold 0.9). E3 detection rate = 0.6 (threshold 0.9).
- Failing seeds in E1 selected an incorrect Arrhenius theta, introducing spurious support terms under 5% multivariate noise.
- Failing seeds in E3 did not admit the missing interaction under the finite-pool validation admission rule.

