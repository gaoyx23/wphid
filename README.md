# WPHID foundation benchmark

This repository contains the minimum temporal weak-form benchmark for
physics-guided interaction discovery. It tests two questions before extending the
method to a spatiotemporal PDE weak form:

1. Can a declared closed library recover the hidden oxidation support without
   pointwise differentiation (E1)?
2. Can cross-condition residual evidence admit one interaction deliberately
   omitted from the initial library (E3)?

The legacy `Phase1/` material is unchanged. New code is packaged under
`src/wphid/`.

## Environment and installation

The final gate was reproduced on Windows with:

- Python 3.11.0
- NumPy 2.4.4
- SciPy 1.17.1
- Matplotlib 3.10.8
- pytest 9.0.3

From the repository root, install the package in editable mode with test
dependencies:

```text
python -m pip install -e ".[test]"
```

Runtime dependencies are NumPy and SciPy. The test extra adds pytest.

## Architecture and leakage boundary

`wphid.truth` generates deterministic observation-only trajectories. The public
trajectory contains `t`, `T`, `pO2`, `sigma`, `Df`, `Do`, and a condition ID; it
does not contain truth coefficients or support labels. `wphid.grammar` constructs
a finite, dimensionally filtered candidate library. `wphid.weak` projects each
condition into compact-support temporal weak integrals. `wphid.sparse` performs
non-centred scale-invariant STLSQ and maps the selected coefficients back to
physical units.

`wphid.pipeline.discover_closed` fits the closed-library E1 model after the
train/validation condition split. `wphid.discovery.discover_with_expansion` ranks
E3 residual candidates using training conditions only, then consults the held-out
condition for admission. Pointwise derivatives are not used for fitting.

Truth support and truth coefficients are evaluation-only: they are used after
inference to calculate support and coefficient errors for the gate report. They
must never be supplied to discovery APIs, threshold selection, residual ranking,
or candidate admission.

## Exact verification commands

Run the complete test suite:

```text
python -m pytest -q
```

Regenerate the clean, closed-library E1 metrics:

```text
python experiments/e1_closed_library.py
```

Regenerate the missing-interaction E3 metrics for seeds 11, 23, 47, 89, and 131:

```text
python experiments/e3_missing_interaction.py
```

E1 applies 5% independent Gaussian noise to every observed field
(`T`, `pO2`, `sigma`, `Df`, `Do`). E3 applies independent zero-mean Gaussian
noise only to the observed `Do` target, with per-condition scale
`1e-4 * max(abs(clean Do))`; its `t`, `T`, `pO2`, `sigma`, and `Df` inputs
remain unchanged.

Regenerate the deterministic gate report from JSON metrics:

```text
python experiments/foundation_report.py
```

## Artifact policy and schema

The reproducible, version-controlled outputs are:

- `artifacts/e1_closed_library/metrics.json`: E1 configuration, seed, selected
  threshold, support, physical coefficients, and per-condition/held-out errors.
- `artifacts/e3_missing_interaction/metrics.json`: E3 aggregate gate metrics and
  all per-seed records.
- `artifacts/e3_missing_interaction/per_seed/seed_<seed>.json`: one compact record
  per E3 seed, including noise scope, fixed/expanded fits, residual evidence,
  admission, false interactions, and gate booleans.
- `docs/results/wphid-foundation-gate.md`: the human-readable final decision and
  exact evidence derived from those JSON metrics.

Large arrays, NPZ payloads, and generated plots under `artifacts/` are ignored.
Compact JSON metrics and Markdown summaries remain eligible for version control.
