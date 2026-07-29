# WPHID Research Implementation Roadmap

This roadmap decomposes the approved WPHID design into independently reviewable plans.

1. **Foundation and discovery gate** — package structure, simulator/discovery isolation, weak-form ODE regression, physics grammar, closed-library recovery (E1), and missing-interaction discovery (E3).
2. **Spatiotemporal PDE weak form** — compactly supported space-time test functions, boundary terms, axisymmetric operators, and PDE noise benchmarks.
3. **Unified explicit–implicit inference** — alternating sparse structure selection and nonlinear parameter estimation for \(E_a,m,n,q\).
4. **Stability and uncertainty** — hierarchical resampling, inclusion probabilities, intervals, and threshold selection without test leakage.
5. **Full validation matrix** — redundant libraries, unknown function families, colored noise, missingness, cross-condition and cross-geometry generalization, baselines, and ablations.
6. **Synthetic telemetry–NDT protocol and paper** — engineering observation emulator, final figures, reproducible experiment manifests, and manuscript rewrite.

Each plan must end in a working, tested deliverable. Plan 2 must not start until Plan 1's E3 gate passes. If E3 fails, the paper scope changes from hidden-interaction discovery to physics-constrained weak-form coupling identification.

