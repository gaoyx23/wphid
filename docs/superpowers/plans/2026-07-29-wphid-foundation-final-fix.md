# WPHID Foundation Final-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the WPHID temporal foundation gate scientifically leakage-safe,
dimensionally enforced, automatically reproducible, and explicitly limited to
E1/E3 finite-pool evidence.

**Architecture:** Public contracts enforce dimensions and result integrity.
E1 uses train-only Arrhenius profile selection and five-seed 5% multivariate
observation noise. E3 uses disjoint screening, finite-pool validation admission,
and test partitions. Experiment modules generate JSON; one renderer derives the
Markdown decision from those JSON artifacts.

**Tech Stack:** Python 3.11, NumPy, SciPy, pytest, setuptools.

## Global Constraints

- Keep `Phase1/` unchanged.
- Never pass truth support, coefficients, or theta to discovery.
- Select theta using training conditions only.
- Use validation only for sparse-model selection or finite-pool admission.
- Evaluate E3 on an untouched test partition after selection.
- Gate E1 at 5% noise on minimum per-seed support F1 >= 0.90.
- Name E3 admission exactly “finite-pool validation admission.”
- Do not claim E2 or full design coverage.
- Generate all numeric report values from persisted artifacts.

---

### Task 1: Enforce result and equation-dimension contracts

**Files:**

- Create: `src/wphid/dimensions.py`
- Modify: `src/wphid/types.py`
- Modify: `src/wphid/grammar/atoms.py`
- Modify: `src/wphid/grammar/generate.py`
- Modify: `src/wphid/pipeline.py`
- Test: `tests/test_types.py`
- Test: `tests/grammar/test_generate.py`
- Test: `tests/integration/test_e1_closed.py`

**Interfaces:**

- Produces: `add_dimensions`, `subtract_dimensions`,
  `validate_equation_dimensions`.
- Extends: `Candidate(..., coefficient_dimension=None,
  canonical_identity=None)`.
- Changes: `discover_closed(..., target_dimension=...)`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_discovery_rejects_feature_without_declared_coefficient_units():
    with pytest.raises(ValueError, match="coefficient dimension"):
        discover_closed(..., target_dimension=DAMAGE_RATE)

def test_discovery_result_rejects_out_of_range_support():
    with pytest.raises(ValueError, match="support"):
        DiscoveryResult(("x",), np.ones(1), np.array([1]), 0.1, {})
```

- [ ] **Step 2: Run the focused tests and confirm they fail for missing
  validation**

Run:
`python -m pytest tests/test_types.py tests/grammar/test_generate.py tests/integration/test_e1_closed.py -q`

Expected: failures show undeclared units are accepted and malformed result
support is accepted.

- [ ] **Step 3: Implement the minimal contracts**

Validate candidate/coefficient/target dimension sums before weak projection.
Generate canonical parameter identities with a shortest round-trip float
encoding. Validate every `DiscoveryResult` field and preserve immutable copies.

- [ ] **Step 4: Run the focused tests**

Run:
`python -m pytest tests/test_types.py tests/grammar/test_generate.py tests/integration/test_e1_closed.py -q`

Expected: pass.

### Task 2: Add train-only theta profile selection

**Files:**

- Modify: `src/wphid/pipeline.py`
- Modify: `experiments/e1_closed_library.py`
- Test: `tests/integration/test_e1_closed.py`

**Interfaces:**

- Produces:
  `discover_profiled_closed(..., candidate_profiles, target_dimension, ...)`.
- Persists: selected theta, canonical profile identity, train-only fold scores,
  and parameter-selection provenance.

- [ ] **Step 1: Write leakage and identity tests**

```python
def test_validation_changes_cannot_change_selected_theta():
    baseline = discover_profiled_closed(...)
    changed = discover_profiled_closed(trajectory_with_changed_validation_Do, ...)
    assert changed.diagnostics["selected_parameter"] == baseline.diagnostics[
        "selected_parameter"
    ]
    assert changed.diagnostics["parameter_profile_history"] == baseline.diagnostics[
        "parameter_profile_history"
    ]
```

- [ ] **Step 2: Run the focused test and confirm the missing API failure**

Run:
`python -m pytest tests/integration/test_e1_closed.py -q`

Expected: fail because profile selection does not exist and E1 fixes theta at
the private value.

- [ ] **Step 3: Implement leave-one-training-condition-out profile selection**

For each declared theta profile and threshold, fit on all but one training
condition and score the held-out training condition. Select theta from the
median fold score plus the fixed sparsity penalty. Only then run ordinary
train/validation sparse selection for the selected profile.

- [ ] **Step 4: Re-run the focused test**

Run:
`python -m pytest tests/integration/test_e1_closed.py -q`

Expected: pass, with parameter diagnostics unchanged after validation-target
perturbation.

### Task 3: Build the genuine 5% multivariate E1 gate

**Files:**

- Modify: `experiments/e1_closed_library.py`
- Test: `tests/integration/test_e1_closed.py`

**Interfaces:**

- Produces: per-seed and aggregate E1 metrics with support
  precision/recall/F1, coefficient errors, selected theta, theta error,
  validation NRMSE, and frozen gate booleans.

- [ ] **Step 1: Write the failing experiment/artifact test**

```python
def test_e1_runs_five_percent_multivariate_noise_and_persists_frozen_metrics(tmp_path):
    metrics = run_experiment(tmp_path)
    assert metrics["configuration"]["observation_noise"]["noise_fraction"] == 0.05
    assert set(metrics["configuration"]["observation_noise"]["applied_to"]) == {
        "T", "pO2", "sigma", "Df", "Do"
    }
    assert metrics["aggregate"]["minimum_support_f1"] >= 0.9
    assert all("coefficient_relative_l2_error" in row for row in metrics["per_seed"])
```

- [ ] **Step 2: Run the focused test and confirm the old clean scalar artifact
  shape fails**

Run:
`python -m pytest tests/integration/test_e1_closed.py -q`

Expected: fail because E1 is clean, single-run, and lacks evaluation metrics.

- [ ] **Step 3: Implement noisy observations and evaluation-only metrics**

Use five fixed noise seeds. Add independent Gaussian noise to every observed
field at `0.05 * RMS(clean field)` per condition. Keep truth values in a
post-discovery evaluator only. Gate on the minimum seed F1.

- [ ] **Step 4: Run the E1 test**

Run:
`python -m pytest tests/integration/test_e1_closed.py -q`

Expected: pass only if the real frozen F1 threshold is met.

### Task 4: Separate E3 screening, admission, and test evaluation

**Files:**

- Modify: `src/wphid/discovery/residual.py`
- Modify: `experiments/e3_missing_interaction.py`
- Modify: `experiments/e3_missing_interaction_core.py`
- Test: `tests/discovery/test_residual.py`
- Test: `tests/integration/test_e3_missing.py`

**Interfaces:**

- Changes:
  `discover_with_expansion(..., train_ids, validation_ids, test_ids,
  target_dimension, ...)`.
- Persists: stage names, finite pool, stability thresholds, residual scores,
  validation-admission evidence, and independent test NRMSE.
- Loads: E1 reference metrics from an explicit JSON path.

- [ ] **Step 1: Write failing split, leakage, and artifact-source tests**

```python
def test_test_target_cannot_change_screening_or_admission():
    baseline = discover_with_expansion(...)
    changed = discover_with_expansion(test_target_perturbed, ...)
    assert changed.diagnostics["expansion_rounds"] == baseline.diagnostics[
        "expansion_rounds"
    ]
    assert changed.diagnostics["test_normalized_rmse"] != baseline.diagnostics[
        "test_normalized_rmse"
    ]
```

- [ ] **Step 2: Run focused E3 tests and confirm failures**

Run:
`python -m pytest tests/discovery/test_residual.py tests/integration/test_e3_missing.py -q`

Expected: fail because no independent test split or artifact-loaded E1
reference exists.

- [ ] **Step 3: Implement the three stages and finite-pool naming**

Reject overlapping split IDs. Rank on training only. Trial the declared finite
pool on validation only. Evaluate fixed, closed-diagnostic, and expanded models
on test only after selection. Load E1 metrics from JSON and delete the copied
NRMSE constant.

- [ ] **Step 4: Re-run focused E3 tests**

Run:
`python -m pytest tests/discovery/test_residual.py tests/integration/test_e3_missing.py -q`

Expected: pass across all five canonical seeds.

### Task 5: Generate the report and declare install dependencies/scope

**Files:**

- Create: `experiments/foundation_report.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/results/wphid-foundation-gate.md`
- Test: `tests/integration/test_foundation_report.py`

**Interfaces:**

- Produces:
  `render_report(e1_metrics, e3_metrics) -> str` and
  `write_report(e1_path, e3_path, output_path)`.
- Declares: NumPy/SciPy runtime and pytest `test` extra.

- [ ] **Step 1: Write failing report-regeneration and metadata tests**

```python
def test_committed_report_equals_renderer_output():
    expected = render_report(load_e1(), load_e3())
    assert REPORT_PATH.read_text(encoding="utf-8") == expected
```

Inspect built project metadata in the test and require runtime dependencies
plus a `test` extra containing pytest.

- [ ] **Step 2: Run the focused tests and confirm failures**

Run:
`python -m pytest tests/integration/test_foundation_report.py -q`

Expected: fail because the renderer and dependency declarations do not exist.

- [ ] **Step 3: Implement deterministic report generation and scope wording**

Every numeric value in the report comes from JSON. State either exact `NO-GO`
evidence or `GO — E1/E3 temporal finite-pool foundation only`; state that E2 is
unimplemented in both cases. Update README commands to
`python -m pip install -e ".[test]"`.

- [ ] **Step 4: Regenerate canonical artifacts and report**

Run:

```text
python experiments/e1_closed_library.py
python experiments/e3_missing_interaction.py
python experiments/foundation_report.py
```

Expected: committed JSON and Markdown are deterministic and internally
consistent.

### Task 6: Full verification, final report, and commits

**Files:**

- Create:
  `.superpowers/sdd/2026-07-29-wphid-foundation-implementation/final-fix-report.md`
- Verify: all modified source, tests, artifacts, and docs.

- [ ] **Step 1: Run full verification**

Run:

```text
python -m pip install -e ".[test]"
python -m pytest -q
python experiments/e1_closed_library.py
python experiments/e3_missing_interaction.py
python experiments/foundation_report.py
python -m pytest -q
```

Expected: install succeeds, tests pass before and after regeneration, and
artifacts remain deterministic.

- [ ] **Step 2: Inspect exact evidence and choose GO/NO-GO**

Read the regenerated aggregate and per-seed metrics. If any frozen threshold
fails, write `NO-GO` and enumerate exact seeds, values, and thresholds. Never
override a failed metric in prose.

- [ ] **Step 3: Write the final-fix report**

Record changed contracts, leakage checks, exact commands/results, artifact
hashes, final decision, E2 boundary, finite-pool boundary, and remaining
concerns. State that stale scratch-ledger claims are superseded without editing
the ignored ledger.

- [ ] **Step 4: Review and commit logically**

Inspect `git diff --check`, the intended file list, and staged diffs. Commit the
approved design/plan separately if useful, then commit implementation plus
generated evidence. Leave no uncommitted intended change.
