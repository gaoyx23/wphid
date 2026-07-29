# WPHID Foundation Final-Fix Design

**Date:** 2026-07-29

**Status:** Approved by the binding final-review findings and the explicit
instruction to execute the final fix wave.

## Scope

This remediation makes the temporal WPHID foundation evidence match the frozen
claims. It does not add the full redundant-library E2 benchmark, spatial weak
forms, arbitrary grammar-neighborhood generation, or broader WPHID modules.
Any successful decision is therefore limited to the E1/E3 temporal finite-pool
foundation. It must not imply E2 or full method-design coverage.

## Scientific boundaries

### E1

E1 will perturb every observed field (`T`, `pO2`, `sigma`, `Df`, and `Do`) with
independent zero-mean Gaussian noise whose standard deviation is 5% of that
field's clean RMS value in each condition. Five fixed noise seeds will be run.
The discovery call receives only noisy trajectories, a declared theta grid,
candidate factories, dimensions, condition splits, and sparse thresholds.

Arrhenius theta is selected by leave-one-training-condition-out profile
selection. Validation observations do not participate in theta selection.
Truth theta, support, and coefficients are read only by post-discovery metric
code. Each seed persists support precision, recall, F1, coefficient relative
L2 error, per-coefficient relative absolute error, selected theta, and theta
relative error. The frozen E1 gate is the minimum per-seed support F1, which
must be at least 0.90.

### E3

E3 is a **finite-pool expansion benchmark**, not a generated residual
neighborhood or open-ended residual-localization method. Its conditions have
three disjoint roles:

1. training-only cross-condition residual screening;
2. **finite-pool validation admission**;
3. independent held-out test evaluation after all selection is complete.

The finite pool is generated once from the declared bounded grammar before
residual screening. Residual correlations, sign consistency, stability
thresholds, rank, score, validation reductions, and test errors are persisted.
Test observations cannot affect screening, ranking, or admission.

E3 loads its E1 reference from the generated E1 JSON artifact. No E1 metric is
copied into source code.

## Contracts and dimensions

`Candidate` retains its raw feature dimension and gains:

- a canonical expression identity that includes a shortest round-trip
  representation of nonlinear parameter values; and
- an optional coefficient dimension.

Discovery requires an equation target dimension. Before constructing any weak
matrix, every candidate must declare coefficient units and satisfy:

`candidate dimension + coefficient dimension == derivative target dimension`.

This allows pressure and pressure-squared features in one equation while
preventing accidental unit mixing or an undeclared dimensional coefficient.

`DiscoveryResult` validates unique non-empty names, finite one-dimensional
coefficients, sorted unique integer support indices within bounds, a finite
non-negative error, and mapping diagnostics.

## Artifacts and decision

The E1 and E3 runners own all numeric metric generation. A report renderer
reads their JSON and deterministically writes
`docs/results/wphid-foundation-gate.md`. Tests regenerate artifacts and the
report in temporary directories and require byte-for-byte report consistency
with the committed report after a canonical gate run.

The final decision is:

- `GO — E1/E3 temporal finite-pool foundation only` when every frozen E1 and
  E3 gate passes; or
- `NO-GO` with the exact failing seeds, values, and thresholds otherwise.

In both cases, the report states that E2 is unimplemented and that no full
WPHID-design coverage claim is supported.

## Verification

Verification consists of focused red/green tests, the full pytest suite,
`python -m pip install -e ".[test]"`, canonical E1 and E3 artifact
regeneration, deterministic report regeneration, artifact/report consistency
tests, and a clean Git status check before committing.
