# P1 check_auto_residual receipt checkpoint

Status: local candidate a444181ae measures X2 299/317; bounded independent
review passed. All eight local acceptance gates are met. No push, merge or main-row admission.

## 1. Goal

Close exactly `postfit/POSTFIT-SURFACE-check_auto_residual` with a reproducible
public-call receipt. Keep the five iSDM rows in a separate change.

## 2. Implemented

Reused the bounded first-seven runners and derivation tools from the retained
candidate. The branch is stacked on the local C6 recovery checkpoint. Retained
R output identifies the pinned P1 oracle; it was not rerun. A fresh Julia process
ran the six fixture cases and their negative controls at commit
`7a2b8dcbe074412f1a1a0c6be0303ed4a1cf8f9f`.

The finalizer binds source, runner, fixture and raw-output hashes. The selected
sidecar adds one residual receipt and its behavior-equivalence entry. Previewing
all receipts confirmed that only this case changed. No iSDM sidecar was admitted.
Assembly changed exactly the same source ID among 317 rows. The C6 reverse-gap
output remained byte-identical to the inherited C6 parent, not to main.

## 3a. Decisions and Rejected Alternatives

Historical Julia captures reference a local-only commit. A fresh capture avoids
making that old commit a public evidence dependency. The historical R output is
retained with an origin manifest and hashes. The five iSDM admissions remain
separate even though the common capture validates all six raw cases.

## 4. Files Touched

Three capture/derivation runners; the behavior and postfit receipt tools; focused
controls; raw evidence and runner snapshots; the selected receipt and sidecar;
the behavior-equivalence table, postfit map, scoreboard and assembled map. No
engine, formula, package version or numerical tolerance changed.

## 5. Checks Run

The fresh Julia capture exited zero in 49.13 seconds under a two-minute hard
stop, with four Julia threads and one BLAS/OMP thread. The focused behavior
suite passed 36/36 controls. The owning behavior tool reports 27 receipts and
24 classes current; these contain 58 behavior entries. The postfit tool reports
seven twin rows and 52 rows current. Assembly checks all 317 rows current.
All five runnable acceptance checks passed re-verification. Exact committed-source
measurement at a444181ae is 299/317; independent review verified the final aggregate.

Full Julia package/core suites, quality tools and Documenter were not run for
this tooling checkpoint. Whole-P1 regression and completion gates remain open.

## 6. Tests of the Tests

Controls reject mismatched labels, fixture identities, raw hashes, source and
runner identities, stale derivation snapshots, and wrong rows. The existing
57 behavior entries plus exactly the residual entry are required; substituting
an iSDM row cannot satisfy the selected-row check.

## 7a. Issue Ledger

No issue was opened, closed or commented on. No new classification signature,
row-admission approval or merge authority was inferred.

## 8. Consistency Audit

The raw finalizer, owning receipt derivation and postfit map agree on the
selected source ID. Independent slice review passed for exact row scope, raw provenance and the
supplemental environment snapshot hashes. This is not the final
P1 completion panel.

## 9. What Did Not Go Smoothly

The initial capture stopped before fitting because this new worktree lacked a
resolved dependency environment. Its failed logs and status are retained. The
working C6 manifest was reused only after exact Project.toml equality; both
hashes are recorded separately. The retry passed without changing source.
A write preview initially compared receipt count with behavior-entry count and
stopped before writing. The corrected preview checks both separately.

## 10. Known Residuals

The candidate has not changed main. Last verified main remains X2 298/317.
The five iSDM rows are implemented separately in candidate 7670e106b, whose
full seven-row checkpoint measures 304/317. C6 timeout recovery is complete.
Remaining scientific rows, F gates, full regression, final review and landing
remain open. The environment manifest
and project are retained as hash-attested raw snapshots. The canonical sidecar
does not independently revalidate that dependency lock; this is not a fully
frozen cross-platform environment.

## 11. Team Learning

A raw capture can cover several cases while a receipt change admits only one.
Count receipts and behavior entries separately, and compare the exact changed
source IDs before writing generated outputs.

## 12. Cross-Product Coverage

Covers one public-door behavior receipt and its invalid-evidence controls.
This checkpoint does NOT cover numerical fit parity, five iSDM admissions, whole-P1 closure,
a full-suite pass, a main result or a public parity claim.

Benchmarks: N/A; no hot-path change. No numerical fit-parity claim is made.
JET, Allocs and Aqua were not rerun for this receipt-tooling slice.
Rose whole-P1 verdict: PENDING final completion panel and landing.
