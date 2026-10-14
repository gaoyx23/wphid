# WPHID reproducibility boundary

`canonical_manifest.json` is the current paper-facing inventory. It records
SHA-256 digests for both positive results and scientifically necessary
limitation results, together with their primary experiment entry points,
supporting physical-model code, protocol hashes, decisions, Python/package
versions, and the source Git commit. The current inventory extends through
R165, including the Bartz thermal audit, engineering-observability audits,
held-out geometry/material/initial-state domain shifts, the local
axisymmetric thermoelastic audit, the qualified external full-bell CalculiX
model-form discrepancy, dual gas-transport discretizations, and the frozen
physical-variation confirmation. The last confirmation is deliberately
recorded as 14/15 accepted groups rather than as a universal pass: geometry
variation under AR(1) noise has a 2.25% wrong-only rate against a 2% ceiling.
R111 additionally records the complete 45-case bounded orthotropy--attachment
sensitivity panel; no best member is promoted to truth.
R112 records four explicit flange/contact controls and a joint-mesh refinement.
It is a public-topology archetype rather than an HM7 manufacturing model, and
is deliberately not promoted to the mechanism-identification truth source.
R113 adds a literature-scale pressure-fixture benchmark using public DLR
specimen dimensions, load levels and orthotropic properties while retaining
the unavailable CT thickness/layup fields as explicit uncertainty.
R114 reuses the normalized R110--R112 fields without new FEA, excludes mesh
and parameter-family duplicates from voting, and records which mechanism
claims remain structurally robust or must be structural-model conditional.
R115--R117 preserve the failed strong-directional development chain: a
constitutively tied flux model improves the strong-isotropic set but erases the
strong-directional control; a clean-field diagnostic locates near-collinearity
before measurement noise; direction-balanced passive weak windows do not cure
the alias. Confirmation remains sealed and R58 stays the canonical safe rule.
R118 then shows that the constitutive grammar uniquely separates both strong
topologies when both flux directions are visible. R119 compresses this oracle:
outer-normal leakage alone systematically aliases the directional truth, while
seven meridional boundary stations plus outer-normal leakage retain unique
recovery through 5% synthetic flux noise. The added meridional channel remains
an information target, not a validated hot-fire sensor, so confirmation remains
sealed and R58 is still the paper-facing rule.
R120 tests global and within-condition centering on the same hard topology
case. Neither restores the directional truth at either weak scale and the
maximum centred topology-column cosine remains above 0.999998, so centering is
kept as a failed sensitivity audit rather than promoted into the physical weak
balance.
R121 maps four conditioned transport candidates through an independent gas
solver to exit spectroscopy and inner-surface microport signals; neither tier
retains the strong-directional truth. R122 locates the loss before regression.
R123 then removes the frozen-driver limitation by qualifying an optional fully
coupled directional Darcy--Forchheimer closure in the seven-field solver. The
feedback reaches carbon and crack, but the strongest declared engineering
block remains only 0.310 noise RMS, so no identification confirmation is run.
R124 reuses these trajectories without a new solve. It shows that spatial or
temporal differencing does not increase the direction signal, while a
near-throat product-gas microport is the only useful member of the uniform
five-port layout. Its three-condition RMS is 0.693 noise units and the selected
high-steam development trace is 1.059; the resulting focused layout remains a
development requirement and needs fresh confirmation.
R125 applies that frozen location to the nominal 900 s trajectories that were
excluded from the placement search. Its RMS is 0.826 and its maximum readout is
0.964 noise units, so the predeclared threshold is not met and no noisy
confirmation is consumed.
R126 then recomputes both strong topologies under the four R100 geometry,
material, initial-defect and combined shifts while keeping the high-steam
focused observation frozen. Three of four shifts exceed one noise RMS; the
material-only shift reaches 0.879, so noisy confirmation remains withheld.
R127 reuses both endpoints and adds only six one-factor trajectories. The
through-thickness conductivity endpoint alone lowers the focused signal to
0.846 noise RMS and explains about 118% of the all-property attenuation, while
density is negligible and the axial-conductivity endpoint partly compensates.
R128 adds three fixed quartile nodes along that single nuisance axis. The five
RMS values decrease monotonically from 1.059 to 0.846 noise units and bracket a
linearly interpolated one-noise boundary at 9.753 W/(m K), providing a frozen
profile for subsequent candidate-forward scoring.
R129 adds the ten missing Darcy trajectories and profiles all four structures
over five material nodes. R130 reuses the resulting twenty-template library to
show that passive multi-channel blocks raise clean unique structure recovery
from 1/20 to 12/20. R131 freezes a 1.20-fold product-microport precision
requirement, at which all 20 clean structure-material truths are unique; noisy
confirmation remains a separate, unused panel.
R132 consumes fresh confirmation seeds under iid, AR(1), and mixed block-bias
noise after noise-only calibration. Coverage remains about 95% with almost no
wrong-only output, but unique structure rates fall to 47.0%, 26.0%, and 24.5%;
the noisy unique-structure claim is therefore withheld.
R133 keeps those confirmation seeds sealed and compares four previously
declared passive observation tiers on disjoint development seeds. The focused
three-species tier is selected, but its worst-noise unique rate is only 33.5%,
so no new confirmation panel is authorized.
R134 replaces the maximum-block rule with covariance-aware joint evidence;
R135 shows that up to four independent passive repeats remain insufficient.
R136 uses fresh development seeds, nested 4/8/12/16 repeats, and a conservative
99% calibration; sixteen repeats are the first passing budget. R137 freezes
that contract and confirms 84.0%--95.75% unique recovery, 99.0%--99.5% truth
coverage, and 0.5%--1.0% wrong-only output across three noise families.
R138 adds sixteen full-coupled midpoint material truths and shows that the
discrete five-node profile fails, especially for Forchheimer structures. R139
develops piecewise-linear continuous profiling without loading those truths;
leave-one-node reconstruction is accurate but sixteen repeats are insufficient.
R140 selects 24 repeats on fresh on-grid development seeds and stops before the
predeclared 32-repeat ceiling. R141 then confirms the frozen rule on R138 truths:
unique recovery is 81.88%--91.88%, coverage 99.06%--99.69%, and wrong-only
output 0.31%--0.94% across the three noise families.
R142 re-solves the gas subsystem with an independent central-difference/Heun
operator and shows that a primary-only candidate library misclassifies both
Forchheimer truths even without measurement noise. R143 restores clean adequacy
by profiling primary and independent gas-operator branches. R144 completes the
dual-solver five-node library and passes all six noise-by-truth-solver groups.
R145 freezes that rule on independent-operator off-grid truths: unique recovery
is 84.69%--93.13%, coverage 99.38%--99.69%, and wrong-only output
0.31%--0.63%. Structural drivers remain primary-solver fields.
R146--R147 reuse the declared geometry/material/defect transfer envelope and
show that measured-geometry conditioning removes a deterministic topology
alias: clean recovery rises from 24/30 to 30/30 without loosening thresholds.
R148--R154 then separate failed threshold reshaping from genuine data
sufficiency. The frozen development choice is 48 passive readouts, the focused
three-species tier, continuous material profiling, the two gas-operator
branches, and a 99.5% global gap calibration. R155 performs 5,000 calibration
draws per geometry--noise pair and 6,000 fresh confirmation identifications.
Fourteen of fifteen predeclared physical-case--noise groups pass; shifted
geometry under AR(1) records 97.75% coverage, 97.5% uniqueness, and 2.25%
wrong-only output, exceeding the 2% ceiling by one of 400 cases. This failure is
retained and the all-group process-robust claim is withheld. R156 and R157 then
use fresh development seeds to test, respectively, fully structure-conditional
thresholds and a narrow directional-candidate guard. The former reduces the
shifted-geometry AR(1) error but drops reference/initial-defect uniqueness to
70.0%/72.75%; the latter restores those groups but leaves shifted-geometry
AR(1) at 2.25%. Neither branch is promoted or confirmed.
R158 changes the evidence rather than the threshold by deriving deterministic
three-species weights from clean candidate-template topology separation. The
weights collapse to an effective 1.19 channels for shifted-geometry AR(1).
On paired fresh development data the equal-GLS control passes 15/15 groups,
whereas weighted GLS passes only 7/15 and worsens the key wrong-only rate from
1.75% to 2.75%. The weighting branch is rejected without a weight search.
R159--R161 then close the previously explicit Darcy-truth gap. Six new
seven-field Darcy trajectories cover material, initial-defect, and combined
shifts; four reference/geometry trajectories are reused. The 30-case clean
gate, 6,000-case development panel, and 6,000-case frozen confirmation are all
uniquely correct. This does not erase the separate Forchheimer AR(1) failure.
R162 then audits within-readout physical-state drift rather than treating all
48 passive readouts as one static state. Across three endpoint paths, four
structures, three covariance scores, and five maximum drift fractions, all
180 clean cases remain uniquely correct. Intermediate observations are linear
paths between fully coupled endpoints, so this authorizes noisy development
but is not yet a nonlinear path or noisy confirmation claim.
R163 adds 21,600 fresh noisy identifications at 0%, 10%, and 25% maximum
within-readout drift. The preselected 10% condition passes all 18 groups and
authorizes one frozen confirmation. The 0% control and 25% stress condition
each fail one shifted-geometry Forchheimer/AR(1) group at 3.25% and 2.25%
wrong-only rates, respectively; the development result is therefore treated
as near-boundary rather than as a monotone robustness claim.
R164 freezes the 10% condition before opening new seeds and confirms all 18
axis--family--noise groups across 7,200 cases. The worst coverage, uniqueness,
and wrong-only rates are 98.25%, 88.75%, and 1.75%. This confirmation does not
extend to the 25% stress condition or to independently solved nonlinear drift
paths.
R165 then performs a no-new-forward-run tangent preflight for measured
geometry metadata error. All 60 clean cases remain unique at 1% of the frozen
reference-to-shift geometry vector, whereas 5% and 10% produce Forchheimer
topology failures. Because candidate fields are interpolated and several
geometry parameters move together, R165 authorizes exact one-factor geometry
candidates but is not a metrology-robustness confirmation.
R166 replaces that tangent approximation with 40 new seven-field candidate
solves and 40 independent-gas counterparts, perturbing length, wall thickness,
expansion ratio, and exit angle one at a time. The exact-geometry control and
the length, thickness, and angle perturbations are unique in all 6 clean
topology--covariance cases. Only the 52.0 to 52.525 expansion-ratio mismatch
fails once, mapping a directional Forchheimer truth to the isotropic topology
under IID scoring. This is an attribution result, not noisy robustness; it
requires expansion ratio to become a profiled geometry nuisance before noise
is added.
R167 then reuses the exact 52.0 and 52.525 candidate endpoints, adds only six
full-coupled truths at the 25%, 50%, and 75% interior ratios, and profiles the
geometry on a frozen 21-point bracket grid together with material and solver
nuisances. Including both exact endpoints gives 30 paired clean cases per
method: treating 52.525 as exact repeats the one R166 topology error, whereas
bracket profiling is unique in 30/30 cases and selects the correct geometry
grid point in every case. This authorizes recalibrated noisy development; it
does not yet confirm robustness under noisy metrology.
The first R168 execution accidentally inherited the 24-readout constant from
R144. It is retained as an explicitly labelled diagnostic: fixed reported
geometry passes 16/30 predeclared groups, while bracket profiling passes 18/30,
raises worst coverage from 54.5% to 97%, and leaves worst uniqueness at 49.5%
and maximum wrong-only rate at 3%. It cannot decide the planned 48-readout
method. The corrected R168 protocol therefore performs a fresh 48-readout
calibration and development run. Only if that corrected run fails may R169 test
the single predeclared 96-readout ceiling; if it passes, R169 is skipped and
R170 reuses the frozen thresholds on disjoint confirmation seeds.

The corrected R168 is now complete: profiling passes 29/30 groups, with worst
coverage 97.5%, worst uniqueness 80.5%, and maximum wrong-only rate 2.5%.
The sole failing stratum is directional Forchheimer at bracket fraction 0.25
under mixed block bias (5/200 wrong). R169 has two completed checkpoints:
IID passes 9/10 groups and AR(1) passes 10/10. Its IID calibration threshold
is zero and the endpoint directional stratum has 6/200 wrong decisions.
The remaining mixed-bias branch is being resumed; the saved failure already
precludes confirmation. R172 audits these saved scores without any new draws
or threshold changes. R170 and R171 are prepared scripts, not completed
experiments. The interrupted R169 worker status is labelled `interrupted`.
The resume run reuses completed noise checkpoints and saves calibration gaps,
RNG state, and development rows atomically every 100 samples.

Regenerate it from the repository root with:

```powershell
python experiments/build_nozzle_truth_architecture_figure.py
python experiments/build_domain_shift_evidence_figure.py
python experiments/build_paper_assets.py
python experiments/paper_r109_external_fea_contract_preflight.py
python experiments/paper_r110_external_axisymmetric_fea_audit.py
python experiments/paper_r111_orthotropic_compliant_attachment_sensitivity.py
python experiments/paper_r112_explicit_sliding_flange_archetype.py
python experiments/paper_r113_dlr_pressure_fixture_benchmark.py
python experiments/paper_r114_structural_model_uncertainty_audit.py
python experiments/paper_r115_constitutively_tied_forchheimer_topology.py --stage development
python experiments/paper_r116_strong_directional_information_diagnostic.py
python experiments/paper_r117_direction_balanced_weak_windows.py
python experiments/paper_r118_direction_resolved_flux_capability.py
python experiments/paper_r119_sparse_boundary_flux_information.py
python experiments/paper_r120_centering_conditioning_audit.py
python experiments/paper_r121_candidate_forward_engineering_observability.py
python experiments/paper_r122_engineering_signal_attribution.py
python experiments/paper_r123_full_coupled_transport_truth.py
python experiments/paper_r124_passive_observation_requirement.py
python experiments/paper_r125_heldout_nominal_focused_observation.py
python experiments/paper_r126_focused_observation_transfer.py
python experiments/paper_r127_material_driver_attribution.py
python experiments/paper_r128_through_conductivity_profile.py
python experiments/paper_r129_material_profiled_candidate_forward.py
python experiments/paper_r130_multichannel_material_profiled_adequacy.py
python experiments/paper_r131_product_precision_contract.py
python experiments/paper_r132_correlated_noise_confirmation.py
python experiments/paper_r133_noise_aware_tier_development.py
python experiments/paper_r134_covariance_aware_joint_scoring.py
python experiments/paper_r135_passive_replication_budget.py
python experiments/paper_r136_conservative_replication_development.py
python experiments/paper_r137_replicated_joint_scoring_confirmation.py
python experiments/paper_r138_offgrid_material_confirmation.py
python experiments/paper_r139_continuous_material_profile_development.py
python experiments/paper_r140_continuous_profile_replication_budget.py
python experiments/paper_r141_offgrid_continuous_profile_confirmation.py
python experiments/paper_r142_independent_gas_operator_preflight.py
python experiments/paper_r143_solver_ensemble_clean_adequacy.py
python experiments/paper_r144_full_solver_ensemble_development.py
python experiments/paper_r145_independent_offgrid_solver_ensemble_confirmation.py
python experiments/paper_r146_process_variation_clean_transfer.py
python experiments/paper_r147_geometry_conditioned_candidate_forward.py
python experiments/paper_r148_noisy_process_variation_confirmation.py
python experiments/paper_r149_structure_conditional_threshold_development.py
python experiments/paper_r150_topology_channel_attribution.py
python experiments/paper_r151_process_robust_replication_development.py
python experiments/paper_r152_conditional_32repeat_development.py
python experiments/paper_r153_48repeat_process_robust_development.py
python experiments/paper_r154_conservative_48repeat_development.py
python experiments/paper_r155_process_robust_confirmation.py
python experiments/paper_r156_conditional_48repeat_development.py
python experiments/paper_r157_directional_guard_development.py
python experiments/paper_r158_physics_weighted_channel_development.py
python experiments/paper_r159_darcy_process_variation_clean_transfer.py --workers 4
python experiments/paper_r160_noisy_darcy_process_variation_development.py
python experiments/paper_r161_darcy_process_variation_confirmation.py
python experiments/paper_r162_readout_state_drift_clean_gate.py
python experiments/paper_r163_noisy_readout_state_drift_development.py
python experiments/paper_r164_readout_state_drift_confirmation.py
python experiments/paper_r165_geometry_metadata_uncertainty_preflight.py
python experiments/paper_r166_exact_single_parameter_geometry_attribution.py --workers 4
python experiments/paper_r167_expansion_ratio_profile_clean_gate.py --workers 4
python experiments/paper_r168_noisy_expansion_ratio_profile_development.py --workers 3
python experiments/paper_r169_96repeat_geometry_profile_budget.py --workers 3
python experiments/paper_r170_geometry_profile_confirmation.py --workers 3
python reproducibility/build_manifest.py
python -m pytest -q
```

Long experiments should be launched through the detached runner so an
interrupted chat does not terminate the scientific process. For example:

```powershell
& .\reproducibility\run_experiment_background.ps1 `
  -Script experiments/paper_r162_readout_state_drift_clean_gate.py `
  -JobName r162-example
```

Each job writes `launcher.json`, `status.json`, `stdout.log`, and `stderr.log`
under `artifacts/_background_jobs/<job-name>/`. Before rerunning an interrupted
experiment, inspect that directory and the experiment's own checkpoint or
final artifact. Detached execution survives chat interruption while the host
machine remains running; it cannot survive a machine shutdown or forced
process termination. Reusing a job name is rejected, which protects against
accidentally launching the same experiment twice. Long experiments should
also checkpoint inside their own artifact directory, so a machine shutdown
requires resuming only unfinished blocks rather than repeating completed
blocks. Background-job control files and logs are local runtime state and are
therefore ignored by Git; the experiment's compact final JSON remains part of
the reproducibility manifest.

Figures 10--14 are emitted by their corresponding R94, R95, R97, R98, and R99
experiment entry points. Re-run those scientific experiments only when their
frozen result payloads are intentionally being replaced; ordinary manuscript
rebuilds merely hash-inventory the existing figure files.

Figure 16 and its metrics are emitted by R108. It reads frozen R100 fields and
does not regenerate the forward trajectories.

R109 freezes the external mesh/load/result contract and analytic limits. R110
runs CalculiX 2.23, reads the same frozen R100 snapshot, and does not regenerate
the forward trajectories. Raw solver traces can contain timestamps and are not
used as byte-reproducibility locks; generated decks and normalized CSV results
are hash-locked in the R110 metrics.

The unchanged official SINDy-PI baseline uses a generated, isolated PySINDy
1.7.5 compatibility environment. The official wheel is retained under
`reproducibility/vendor/`; rebuild the ignored local environment only when the
baseline must be rerun:

```powershell
powershell -ExecutionPolicy Bypass -File reproducibility/setup_official_sindypi.ps1
```

The following are deliberately outside the current manifest:

- worker shards under `artifacts/**/workers/`;
- development-only tuning artifacts;
- superseded development histories under `.retired/2026-09-12/`;
- LaTeX auxiliary files and local compatibility environments.

A release is frozen only when the manifest reports `worktree_clean: true` and
its commit is tagged. Until then the hashes freeze the result payloads but not
the whole repository state.

The latest full-suite verification and targeted follow-ups are stored in
`test_verification.json` and embedded into the generated manifest. Targeted
follow-ups do not replace the recorded full-suite run.
