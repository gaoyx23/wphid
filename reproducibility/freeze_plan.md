# WPHID release-freeze plan

The current workspace contains intentional research changes together with older
file deletions and moved documents whose ownership cannot be inferred safely.
They must not be swept into one blanket commit.

## Commit A — method and tests

- `src/wphid/`
- paper-facing and stage experiment entry points under `experiments/`
- `tests/`
- `pyproject.toml`

Required gate: `python -m pytest -q` must reproduce the pass count recorded in
`test_verification.json`.

## Commit B — evidence and audit

- every metric file listed in `canonical_manifest.json`;
- `docs/paper/`, `docs/experiment-catalog-zh.md`, and project navigation;
- `reproducibility/`.

Required gate: regenerate `canonical_manifest.json` and verify that every
listed SHA-256 is unchanged.

## Commit C — manuscript

- final manuscript source, figures, tables, and supplementary material only.

After Commit C, create one annotated release tag. Retired development
artifacts, local environments, LaTeX auxiliaries, and unrelated root-level
deletions remain outside the release commit unless explicitly reviewed.

This staged sequence allows the immutable release commit to be reconstructed
without discarding or silently adopting unrelated workspace changes.
