# WPHID foundation discovery gate

## Decision

**GO: proceed to spatiotemporal PDE weak form.**

This decision is limited to the declared temporal foundation benchmark and its
finite physics grammar. It does not claim general equation discovery or broad
noise robustness.

## Reproduction evidence

The metrics were regenerated from the repository root with:

```text
python experiments/e1_closed_library.py
python experiments/e3_missing_interaction.py
```

The final test command and result were:

```text
python -m pytest -q
46 passed in 9.01s
```

Environment: Python 3.11.0, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8,
pytest 9.0.3.

## Evaluation definitions

Truth support and coefficients are read only after discovery for evaluation.
Support precision, recall, and F1 compare inferred E1 support with the three-term
hidden support. Coefficient relative L2 error is
`||xi_hat_S - xi_truth_S||_2 / ||xi_truth_S||_2` on that truth support.
Normalized RMSE is `RMS(residual) / RMS(target)`.

## E1: closed library

- Seed: `20260729`.
- Noise level: `0.0` (clean observations).
- Train conditions: `thermal_hold`, `fatigue_rich`.
- Validation condition: `oxygen_rich`.
- Selected threshold: `0.013333378729599095`.
- Support precision: `1.0`.
- Support recall: `1.0`.
- Support F1: `1.0` (required: at least `0.90`).
- Coefficient relative L2 error: `0.0017008460636964463`.
- Validation normalized RMSE: `0.00025879605792407254`.
- Admitted false terms: none.

| Truth-supported term | Truth coefficient | Estimate | Relative absolute error |
|---|---:|---:|---:|
| `arr(T;theta=1800)*pO2` | 0.18 | 0.18009506547914711 | 0.00052814155081731199 |
| `arr(T;theta=1800)*pO2*sigma` | 0.006 | 0.0059992102057036187 | 0.00013163238273024061 |
| `arr(T;theta=1800)*pO2*Df` | 0.45 | 0.44918109648928445 | 0.0018197855793679176 |

The inferred support exactly equals the evaluation support; the other three
declared candidates have zero fitted coefficients.

## E3: interaction omitted from the initial library

- Seeds: `11`, `23`, `47`, `89`, `131`.
- Noise fraction: `0.0001`.
- Noise distribution: independent zero-mean Gaussian per time sample.
- Noise scale: `noise_fraction * max(abs(clean Do))` independently per
  condition.
- Noise scope: target-only `Do` observation noise. `t`, `T`, `pO2`, `sigma`, and
  `Df` are unchanged.
- Expected missing interaction: `arr(T;theta=1800)*pO2*Df`.
- Missing-interaction detection rate: `1.0` (`5/5`; required: at least `0.90`).
- Maximum admitted false interactions: `0`.
- Maximum expanded validation NRMSE / Task 6 E1 validation NRMSE:
  `0.9204354692018799` (required: at most `1.2`).

| Seed | Fixed-library validation NRMSE | Expanded validation NRMSE | Expanded / Task 6 E1 | Expanded coefficient relative L2 error | Admitted interaction | False interactions |
|---:|---:|---:|---:|---:|---|---|
| 11 | 0.024493133723039039 | 0.0002382050710029406 | 0.92043546920187991 | 0.00057473968214486132 | `arr(T;theta=1800)*pO2*Df` | none |
| 23 | 0.024552035048722179 | 0.00014429723856345086 | 0.55757123860745139 | 0.0007577019109553957 | `arr(T;theta=1800)*pO2*Df` | none |
| 47 | 0.024639830113450733 | 0.00017134453828185003 | 0.66208326222697078 | 0.00037373692942956597 | `arr(T;theta=1800)*pO2*Df` | none |
| 89 | 0.024566297728551727 | 0.00021672831550472016 | 0.8374482874399326 | 0.00067250634407154818 | `arr(T;theta=1800)*pO2*Df` | none |
| 131 | 0.02460810942042594 | 0.00022841711191699576 | 0.88261434022310503 | 0.00063413205078611504 | `arr(T;theta=1800)*pO2*Df` | none |

Every seed also passed the Task 7 criteria:

1. the fixed-library validation error is at least 50% above its same-seed
   closed-library diagnostic;
2. residual diagnostics contain structured cross-condition evidence computed
   from training conditions only;
3. expansion admits the canonical missing interaction;
4. expanded validation error is within 20% of the fixed Task 6 E1 reference;
5. no more than one false interaction is admitted.

## Gate rationale and boundary

E1 exceeds the required support-F1 gate, and E3 exceeds the required detection
rate while satisfying every Task 7 criterion with zero false admissions. These
results support proceeding to Plan 2's spatiotemporal PDE weak-form stage.

The next stage must not treat this gate as evidence for noisy predictors or full
field noise. Task 7 perturbs only `Do`, at a small `1e-4` target-relative scale.
The current benchmark is temporal, uses three training/validation condition
families, and searches only inside the declared bounded grammar. Plan 2 should
separately test spatial discretization, boundary handling, field noise,
covariate noise, and out-of-grammar failure modes.
