# WPHID Foundation and Missing-Interaction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested WPHID minimum viable method that recovers known interactions without pointwise differentiation and detects/reconstructs one interaction deliberately omitted from the initial library.

**Architecture:** New code lives in an importable `wphid` package and does not modify the legacy Phase 1 baseline. A hidden truth module generates trajectories and exports observations only; the discovery pipeline receives arrays plus physics metadata, constructs weak-form regressions, generates admissible candidates, performs sparse selection, and expands the library from cross-condition residual evidence.

**Tech Stack:** Python 3.11, NumPy 2.4, SciPy 1.17, Matplotlib 3.10, pytest 9.0; no new runtime dependency in this plan.

## Global Constraints

- Keep `Phase1/` unchanged as the E1 legacy baseline.
- Truth-generation code and discovery code must be in separate modules; discovery APIs must never receive the truth support.
- Scale regression columns without centering and map coefficients back to physical units.
- Split train/validation conditions before threshold or expansion-rule selection.
- Use weak integrals for model fitting; pointwise derivatives may appear only in diagnostic comparisons.
- Interpret “discovery” as finite discovery inside the declared physics grammar.
- Commit after every task; do not combine unrelated tasks.

---

## File Map

- `pyproject.toml` — package and pytest configuration.
- `src/wphid/types.py` — shared immutable data contracts.
- `src/wphid/truth/aging_ode.py` — hidden multi-condition truth generator.
- `src/wphid/weak/time.py` — compact-support temporal weak projection.
- `src/wphid/grammar/atoms.py` — candidate atoms and dimensional metadata.
- `src/wphid/grammar/generate.py` — admissible F/G candidate construction.
- `src/wphid/sparse/stlsq.py` — scaled sparse regression and unbiased refit.
- `src/wphid/discovery/residual.py` — missing-structure diagnostic and controlled expansion.
- `src/wphid/pipeline.py` — end-to-end closed/open-library APIs.
- `experiments/e1_closed_library.py` — reproducible E1 experiment.
- `experiments/e3_missing_interaction.py` — reproducible E3 gate.
- `tests/` — unit and integration tests mirroring package modules.

### Task 1: Package skeleton and data contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/wphid/__init__.py`
- Create: `src/wphid/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Produces: `Trajectory(t, states, condition_id)`, `Candidate(name, evaluate, dimension, group, parents)`, and `DiscoveryResult(names, coefficients, support, validation_error, diagnostics)`.

- [ ] **Step 1: Write the failing contract test**

```python
import numpy as np
import pytest
from wphid.types import Trajectory

def test_trajectory_rejects_mismatched_time_axis():
    with pytest.raises(ValueError, match="time axis"):
        Trajectory(np.arange(3.0), {"Do": np.zeros(2)}, "c0")
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `python -m pytest tests/test_types.py -v`  
Expected: FAIL because `wphid` does not exist.

- [ ] **Step 3: Add package configuration and immutable dataclasses**

Set `pyproject.toml` to use setuptools with `package-dir = {"" = "src"}` and pytest `pythonpath = ["src"]`. Implement `Trajectory.__post_init__` to require one-dimensional strictly increasing `t` and a matching first axis for every state. Define `Candidate.evaluate(states) -> np.ndarray`.

- [ ] **Step 4: Run the contract test**

Run: `python -m pytest tests/test_types.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/wphid tests/test_types.py
git commit -m "build: add WPHID package contracts"
```

### Task 2: Isolated multi-condition truth generator

**Files:**
- Create: `src/wphid/truth/__init__.py`
- Create: `src/wphid/truth/aging_ode.py`
- Test: `tests/truth/test_aging_ode.py`

**Interfaces:**
- Consumes: `Trajectory`.
- Produces: `simulate_condition(condition_id: str, t: np.ndarray, forcing: dict[str, np.ndarray]) -> Trajectory`.
- The exported trajectory contains `T`, `pO2`, `sigma`, `Df`, and `Do`; it does not contain coefficient values or support labels.

- [ ] **Step 1: Write a failing black-box truth test**

```python
def test_conditions_independently_excitate_sigma_and_fatigue():
    low = simulate_named_condition("thermal_hold")
    cyc = simulate_named_condition("fatigue_rich")
    assert low.states["sigma"].max() > cyc.states["sigma"].mean()
    assert cyc.states["Df"][-1] > 2 * low.states["Df"][-1]
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/truth/test_aging_ode.py -v`  
Expected: FAIL because the truth module is absent.

- [ ] **Step 3: Implement three deterministic conditions**

Use `scipy.integrate.solve_ivp` for:

\[
\dot D_f=a_f\Delta\sigma^3(1-D_f)^{-1},\qquad
\dot D_o=a_oA(T)p_{O_2}+a_{os}A(T)p_{O_2}\sigma+a_{of}A(T)p_{O_2}D_f.
\]

Provide `thermal_hold`, `fatigue_rich`, and `oxygen_rich` forcings with independently varying \(T,\sigma,p_{O_2}\). Keep truth coefficients private to this module.

- [ ] **Step 4: Verify trajectory bounds and excitation**

Run: `python -m pytest tests/truth/test_aging_ode.py -v`  
Expected: PASS; all damage values remain in `[0, 0.8)`.

- [ ] **Step 5: Commit**

```bash
git add src/wphid/truth tests/truth
git commit -m "feat: add isolated multi-condition truth generator"
```

### Task 3: Temporal weak-form projection

**Files:**
- Create: `src/wphid/weak/__init__.py`
- Create: `src/wphid/weak/time.py`
- Test: `tests/weak/test_time.py`

**Interfaces:**
- Produces: `project_time_weak(t, y, features, window, stride) -> tuple[np.ndarray, np.ndarray]`.
- `features` has shape `(n_time, n_terms)`; output is `(A, b)` satisfying \(A\xi=b\).

- [ ] **Step 1: Write exact polynomial and noisy exponential tests**

```python
def test_weak_projection_recovers_linear_decay_without_derivatives():
    t = np.linspace(0, 4, 401)
    y = np.exp(-2 * t)
    A, b = project_time_weak(t, y, y[:, None], window=81, stride=20)
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    assert coef[0] == pytest.approx(-2.0, rel=2e-3)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/weak/test_time.py -v`  
Expected: FAIL because the weak module is absent.

- [ ] **Step 3: Implement compact-support weak projection**

Use an even polynomial bump \(\psi(s)=(1-s^2)^4\) on `[-1,1]`, analytic \(\psi_t\), trapezoidal quadrature, and windows wholly inside the observation interval:

\[
b_k=-\int \psi_t y\,dt,\qquad A_{kj}=\int\psi\theta_j\,dt.
\]

- [ ] **Step 4: Verify clean and 5% noisy recovery**

Run: `python -m pytest tests/weak/test_time.py -v`  
Expected: PASS; clean relative error below `0.2%`, median 5% noise error below `8%` over 20 seeds.

- [ ] **Step 5: Commit**

```bash
git add src/wphid/weak tests/weak
git commit -m "feat: add temporal weak-form projection"
```

### Task 4: Physics grammar and dimensional filtering

**Files:**
- Create: `src/wphid/grammar/__init__.py`
- Create: `src/wphid/grammar/atoms.py`
- Create: `src/wphid/grammar/generate.py`
- Test: `tests/grammar/test_generate.py`

**Interfaces:**
- Produces: `aging_atoms() -> list[Candidate]` and `generate_candidates(atoms, target_dimension, max_order=3) -> list[Candidate]`.

- [ ] **Step 1: Write grammar tests**

Test that the grammar generates `arr(T)*pO2*Df` and `arr(T)*pO2*sigma`, rejects dimensionally invalid additions, removes commutative duplicates, and labels candidates as `F`, `G2`, or `G3`.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/grammar/test_generate.py -v`  
Expected: FAIL because grammar modules are absent.

- [ ] **Step 3: Implement atoms and bounded composition**

Represent dimensions as integer exponent tuples over base units. Generate products up to order 3, Arrhenius modulation only with `T`, and declared saturation transforms. Deduplicate by canonical sorted expression trees, not display names.

- [ ] **Step 4: Verify deterministic candidate generation**

Run: `python -m pytest tests/grammar/test_generate.py -v`  
Expected: PASS and identical ordered names across two runs.

- [ ] **Step 5: Commit**

```bash
git add src/wphid/grammar tests/grammar
git commit -m "feat: add physics-guided interaction grammar"
```

### Task 5: Scaled sparse regression

**Files:**
- Create: `src/wphid/sparse/__init__.py`
- Create: `src/wphid/sparse/stlsq.py`
- Test: `tests/sparse/test_stlsq.py`

**Interfaces:**
- Produces: `fit_stlsq(A, b, threshold, max_iter=20) -> DiscoveryResult`.

- [ ] **Step 1: Write scaling and support tests**

Construct a matrix whose columns differ by \(10^8\), with true support `[0, 2]`. Assert exact support and coefficients after physical-scale conversion.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/sparse/test_stlsq.py -v`  
Expected: FAIL because sparse module is absent.

- [ ] **Step 3: Implement non-centred scaling, thresholding, and refit**

Divide each column and target by standard deviation, threshold scaled coefficients, iterate support, refit selected raw columns without regularization, and report condition number plus excluded zero-variance columns.

- [ ] **Step 4: Verify**

Run: `python -m pytest tests/sparse/test_stlsq.py -v`  
Expected: PASS with coefficient relative error below `1e-8`.

- [ ] **Step 5: Commit**

```bash
git add src/wphid/sparse tests/sparse
git commit -m "feat: add scale-invariant sparse regression"
```

### Task 6: Closed-library E1 pipeline

**Files:**
- Create: `src/wphid/pipeline.py`
- Create: `experiments/e1_closed_library.py`
- Test: `tests/integration/test_e1_closed.py`

**Interfaces:**
- Produces: `discover_closed(trajectories, candidates, train_ids, validation_ids, thresholds) -> DiscoveryResult`.

- [ ] **Step 1: Write the integration test**

Assert that training on two conditions and validating on the third recovers all three `Do` terms, no spurious term, and validation normalized RMSE below `0.03` without accessing truth support in `pipeline.py`.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_e1_closed.py -v`  
Expected: FAIL because pipeline is absent.

- [ ] **Step 3: Implement leakage-safe model selection**

Build weak matrices per condition, choose threshold only by validation error plus sparsity penalty, refit on training conditions, and return term names, coefficients, support, validation error, and diagnostics.

- [ ] **Step 4: Run E1**

Run: `python -m pytest tests/integration/test_e1_closed.py -v`  
Expected: PASS.

Run: `python experiments/e1_closed_library.py`  
Expected: writes `artifacts/e1_closed_library/metrics.json` with fixed seed and configuration.

- [ ] **Step 5: Commit**

```bash
git add src/wphid/pipeline.py experiments/e1_closed_library.py tests/integration/test_e1_closed.py
git commit -m "feat: validate closed-library weak discovery"
```

### Task 7: Residual-driven missing-interaction expansion

**Files:**
- Create: `src/wphid/discovery/__init__.py`
- Create: `src/wphid/discovery/residual.py`
- Create: `experiments/e3_missing_interaction.py`
- Test: `tests/discovery/test_residual.py`
- Test: `tests/integration/test_e3_missing.py`

**Interfaces:**
- Produces: `rank_expansions(residuals_by_condition, base_candidates, expansion_candidates) -> list[ExpansionScore]`.
- Produces: `discover_with_expansion(..., max_rounds=2) -> DiscoveryResult`.

- [ ] **Step 1: Write residual-ranking unit test**

Generate residuals proportional to `arr(T)*pO2*Df` plus noise. Assert this candidate ranks first and a candidate correlated in only one condition fails the cross-condition stability rule.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/discovery/test_residual.py -v`  
Expected: FAIL because residual module is absent.

- [ ] **Step 3: Implement controlled expansion**

Score candidates by median absolute weak-residual correlation across training conditions, sign consistency, and validation-error reduction. Admit one complexity layer per round and reject candidates that improve only training error.

- [ ] **Step 4: Write and run the E3 integration test**

Start without `arr(T)*pO2*Df`. Assert:

1. fixed-library validation error exceeds the closed-library result by at least 50%;
2. diagnostics report structured residual evidence;
3. expansion adds `arr(T)*pO2*Df` or its canonical equivalent;
4. expanded validation error returns within 20% of E1;
5. no more than one false interaction is admitted.

Run: `python -m pytest tests/integration/test_e3_missing.py -v`  
Expected: PASS over seeds `11, 23, 47, 89, 131`.

- [ ] **Step 5: Run the gate experiment**

Run: `python experiments/e3_missing_interaction.py`  
Expected: writes per-seed and aggregate metrics to `artifacts/e3_missing_interaction/`.

- [ ] **Step 6: Commit**

```bash
git add src/wphid/discovery experiments/e3_missing_interaction.py tests/discovery tests/integration/test_e3_missing.py
git commit -m "feat: discover interactions missing from initial library"
```

### Task 8: Reproducibility, full verification, and gate report

**Files:**
- Create: `README.md`
- Create: `docs/results/wphid-foundation-gate.md`
- Modify: `.gitignore`
- Test: all tests.

**Interfaces:**
- Produces a documented go/no-go decision for Plan 2.

- [ ] **Step 1: Document exact commands and architecture**

Document environment versions, package installation (`python -m pip install -e .`), test commands, E1/E3 commands, artifact schema, and the rule that truth support is evaluation-only.

- [ ] **Step 2: Ignore generated artifacts except compact metrics**

Add `artifacts/**/arrays/`, `artifacts/**/*.npz`, and generated plots to `.gitignore`; retain JSON metrics and Markdown summaries.

- [ ] **Step 3: Run complete verification**

Run: `python -m pytest -q`  
Expected: all tests pass.

Run: `python experiments/e1_closed_library.py`  
Expected: E1 support F1 at least `0.90`.

Run: `python experiments/e3_missing_interaction.py`  
Expected: missing-mechanism detection rate at least `0.90` and the E3 criteria in Task 7.

- [ ] **Step 4: Write the gate report from generated metrics**

The report must state exact seeds, noise level, support metrics, coefficient error, validation error, admitted false terms, and either:

- `GO`: proceed to spatiotemporal PDE weak form; or
- `NO-GO`: retitle scope to physics-constrained weak-form coupling identification and analyze failure modes.

- [ ] **Step 5: Commit**

```bash
git add README.md .gitignore docs/results artifacts
git commit -m "docs: record WPHID foundation discovery gate"
```

