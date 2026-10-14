# WPHID: mechanism-level joint weak profiling

Research software and saved synthetic evidence for nonlinear multiphysics mechanism identification. WPHID compares declared function families, destination equations and continuous internal-parameter sharing partitions, with equation-specific linear coefficients profiled during bounded nonlinear search.

The retained reference implementation is `src/wphid/final_candidate_v1/solver.py`. This is a dated research reference and conditional workflow, not a fully qualified estimator for arbitrary open structures or all unknown material properties. No health/lifetime prediction is implemented as the study's contribution.

## Evidence and limitations

The corrected 18-noise-block conditional comparison supplies family, active support and partition. WPHID and corrected Adam each pass 18/18 shared, 2/18 partial and 5/18 independent reaction-parameter/amplitude cases; first-order Gaussian weak quasi-likelihood passes 18/18, 0/18, 0/18. The Gaussian shared median score is descriptively 52.44% lower at 44.39 times median solver CPU cost. All predeclared whole-role advantage tests fail. The current fixed workflows are not best-tuned native-method comparisons; the Gaussian program is not official WENDy.

Original separate sharing controls support appropriate sharing; complete one-reaction search selects 9/9 positive structures and partitions but achieves same-fit precision in 5/9. The original eight development packages remain 6/8. Source-coupled prototypes and other failed developments retain their unadopted status. Engineering transport calibration has its own supplied-mechanism and synthetic observation scope.

## Reproduction

Source, fixed protocols, runtime locks and saved-result summary tables are tracked in this repository candidate. Large frozen observation/result archives belong in a dated GitHub Release. The release file manifest lists exact SHA256 and archive names; no release URL or DOI is invented before publication.

1. Extract the independent reproduction ZIP into a new directory.
2. Extract the embedded original V1 and development archives there, preserving their relative paths and checking the archive hashes first. These are preserved dependencies, not newly generated data.
3. Run the provided saved-file verification utility. It checks named hashes and the 216-row/162-fresh-row transcription, without invoking estimators, integrations or model replay.
4. Use the frozen runtime/requirements files for any separately intended computation. Budget-based optimization may follow different histories on different hardware; faithful saved-result verification is a different claim.

The tracked candidate excludes account/session credentials and unreviewed manuscript drafts. Platform paths in frozen runtime/provenance records remain documentary identities rather than active external dependencies. Historical larger arrays absent from the worktree are not regenerated; their availability remains recorded in the preserved catalogue.

## License and attribution

First-party code and associated software documentation use the [MIT License](LICENSE), copyright 2026 Yixuan Gao. Synthetic benchmark arrays and saved estimates are supplied as associated reproducibility fixtures with their provenance; please cite the software and the final article when available. Third-party packages retain their own licenses, described in THIRD_PARTY_NOTICES.md. Author papers are linked, not redistributed or relicensed.

## Citation

See CITATION.cff. The article is an author-review manuscript; no journal acceptance, public DOI or publication metadata is asserted.
