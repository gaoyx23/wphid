# Conditional Gaussian weak-PDE comparison

This directory contains a separately frozen supplemental estimator comparison,
not an execution or a validated PDE port of official WENDy.jl. Scientific output
is written exclusively beneath `artifacts/paper_wendy_comparison_v1`.

The official nonlinear-in-parameters method is described by Rummel et al.,
[arXiv:2502.08881](https://arxiv.org/abs/2502.08881), with author source at
[WENDy.jl](https://github.com/nrummel/WENDy.jl). Nine official files were retrieved
and hashed at commit `edaaab7392adf8b307a4a607c4210a910db41076`, version 1.1.0;
see `official-source-audit-20261013.json` for exact byte hashes and scope.

The official constructor forms temporal ODE weak residuals and estimates one
noise standard deviation per state dimension. It has no public contract for
the archived axisymmetric tensor-space-time weak operators. A method-of-lines
application could be a legitimate adaptation after independent qualification,
but changes spatial/boundary discretization and covariance handling. Its current
dense nonlinear Hessian implementation also allocates two second-derivative
covariance-factor buffers of shape `K*D by (M+1)*D by J by J` (one shaped tensor).
For even one archived condition, `D=1875`, `M+1=25`, `J=15`, and the lower bound
`K=1` require 316,406,250,000 bytes for those two buffers, about 294.68 GiB.
This exceeds the recorded local 15.75 GB memory before other allocations.
These source-level constraints, rather than absence of a Julia executable alone,
prevent claiming an official matched PDE comparison in this batch.

The transparent additional comparator uses the saved PDE weak residual and its
complete first-order raw observation Jacobian. Covariance includes shared state
errors across equations and overlapping weak windows, and depends on all linear
and nonlinear parameters. It minimizes one half of residual Mahalanobis square
plus log determinant, with an analytic gradient including covariance changes.
Conditions are independent noise blocks. It neither assumes independent weak
rows nor regularizes singular covariance. This is a Gaussian quasi-likelihood
based on linearization at observed states; it is not an exact nonlinear noise
likelihood, a calibrated confidence method, or the official author solver.

The fixed known-support experiment reuses 18 archived Q1 observation packages
and 18 complete WPHID controls, with no new noise or trajectory integration.
The comparison conditions on the same Arrhenius family, active destinations and
partition. Six training and two heldout conditions use the original training
normalization. Noise sigmas are 0.0005 times observed training field RMS, never
clean truth. All saved acquisition clipping counts are checked to be zero.
Linear coefficients are signed as in the archived WPHID control. All 12 linear
coefficients and 3, 6, or 9 internal coordinates are optimized jointly. Initial
linear coefficients come from OLS at each predeclared nonlinear fraction;
archived fitted parameters are not supplied as starts.

`scientific-protocol-20261013.json` fixes six seeds, three roles, three starts,
600 CPU seconds and 60,000 objective calls for each complete fitting program.
Numerical failures and budget exhaustion are retained. No new starts, covariance
jitter, row removal, or truth-based score selection are permitted. The protocol,
implementation, tests, runner, archived dependencies, inputs and controls are
hashed before the first fit. Actual software test output and runtime preflight
are retained separately in the artifact directory.

Commands (run once, from the algorithm worktree):

```powershell
python reproducibility/wendy_comparison_v1/run_fixed_comparison.py --stage prepare
python reproducibility/wendy_comparison_v1/run_fixed_comparison.py --stage run
python reproducibility/wendy_comparison_v1/run_fixed_comparison.py --stage assess
```

Preparation and assessment use exclusive output creation. Execution verifies
and skips hash-matched complete records; it never refits a completed experiment.
Truth references are imported for assessment only after all 18 immutable fit
records exist. Parameter precision and numerical qualification are separate
endpoints. This batch supplies conditional development evidence and does not
qualify an open-search estimator or an independent confirmation.
