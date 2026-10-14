# WPHID final candidate: local reproducibility release candidate

This package is prepared for author review. It has not been publicly deposited
or submitted. Data may be made public according to the sole author's recorded
decision; the code license and public identifier remain to be specified.

## Contents and scope

The package retains the independently versioned candidate, the explicitly
conditional ADAM-SINDy adaptation, frozen scientific protocol/source/input
identities, raw new observed fields and supplied manufactured source terms,
weak matrices/operators, saved estimates, assessment, and read-only diagnostics.
The saved-output replay checks equation RSS, common scores, coefficient units
and frozen held-out residuals from archived parameters without fitting.
Terminal ADAM Gamma values are not archived, so the complete adaptive penalty
and optimizer checkpoint are not independently replayable from this package.
The separately predeclared adaptation-scale diagnostic checks whether the
fixed threshold excludes any reference-amplitude precision band; it does not
change the primary results, tune on confirmation or refit an estimator.
It also embeds the 61 historical source records used by the manuscript.
Their original bytes are preserved at their catalogue paths. These historical
records do not reconstruct every large old trajectory array or the historical
execution environments.

The fresh experiment has 18 independent Q1 noise blocks and 3 independent Q3
blocks. Role packages within each block share field noise. They are not
independent repetitions. The earlier cost-development block is separate.
There are no newly integrated physical trajectories in this qualification.

The source freeze precedes confirmation input preparation. All confirmation
inputs were hashed before fitting. Truth assessment follows immutable estimate
records. The execution order is Q1 conditional estimation, Q2 sharing controls,
and Q3 one-reaction full finite search. A process stops at its fixed CPU cap;
the cap does not certify convergence. Invalid estimates and failures remain
in the declared denominators.

Q3 covers 43 candidates for one reaction class, not the three-class 79507-space.
The ADAM branch uses common weak inputs and explicit adaptations described in
the official-source lock. It is not author-native end-to-end ADAM-SINDy or a
fully tuned performance benchmark. New qualification does not requalify every
historical WPHID version or resolve all coexisting-mechanism precision.

## Software environment

Recorded environment: Windows AMD64, Python 3.11.0, NumPy 2.4.4, SciPy 1.17.1,
PyTorch 2.11.0+cu126. Scientific computations use CPU float64 and one numerical
thread per worker. Four worker processes were prospectively fixed on an
Intel Core i9-13900HX with 32 logical processors and 15.75 GB RAM.

`requirements-software-lock.txt` pins the observed dependency closure; the
runtime lock also records the installed distribution snapshot. The installed
PyTorch version has a matching Python 3.11 Windows wheel in the
[official cu126 index](https://download.pytorch.org/whl/cu126/torch/).
An installation can use that index as an extra index for the pinned PyTorch
build. No wheel download hash or fresh-environment installation is claimed.
Another platform/build requires its own software qualification before treating
it as the same numerical implementation.
CPU-time truncation can also end a search at different points on different
processors or under different execution conditions. Recorded seeds therefore
do not certify identical complete optimization histories. Saved-model replay
verifies the numerical outputs of the archived estimates without optimization.

## Verification and reuse

Extract at a location chosen by the reader; preserve relative paths and execute
from that root with `src` ahead of other `wphid` installations. The runtime
checks the actual import and rejects another editable installation. Source and
input bytes must match their SHA256 identities.

The three software test modules contain 43 checks on literal fixtures and do
not generate scientific observations. The saved qualification manifests record
which checks were actually run in the study.

The execution driver intentionally reuses completed success and failure records
and refuses to overwrite a completed experiment. In the study workspace,
completed experiments must not be rerun. Readers wishing to carry out a new
independent replication should create a new version/output protocol rather than
alter this freeze or erase its completed records. Saved numeric records and
input identities can be inspected without new fitting.

The original execution was interrupted after 108 Q1, 36 Q2 and 87 Q3 records
had been saved. A checkpoint-continuation audit binds these 231 original files
by hash before continuing only the 558 absent Q3 jobs. The audit is included
in the candidate archive. Saved worker-job CPU totals omit unsaved work from
the interrupted processes and coordinator/postprocessing overhead; they are
not a complete wall-time or system-energy measurement. At most four Q3 fits
could have been in progress, each subject to the unchanged 30-second optimizer
CPU budget. The GPU permission was recorded separately; the present scientific
confirmation retains its prospectively fixed CPU configuration.

The manuscript's source catalogue, transcription tables and new assessment
allow its reported arithmetic to be checked from saved output. A local ZIP is
a release candidate, not a public repository, DOI, code license, certification,
or author approval of the final manuscript.
