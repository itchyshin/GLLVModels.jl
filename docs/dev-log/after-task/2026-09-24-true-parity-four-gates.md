# After-task: the four true-parity paste gates, executed (2026-09-24)

Lane: Claude, `true-parity-20260924`. Kit: `LOOP/lanes/true-parity-20260924/`. Source handover: `docs/dev-log/handover/2026-09-24-claude-handover.md`.

## 1. Goal

Carry the OWED steps of the 2026-09-24 handover to their runbook stop lines once Shinichi pasted all four gate strings:
- S4 probe (#409)
- Totoro #323 Track A (#410)
- Delta dispersion A (#399)
- D3 Stage 1 (#411)

The handover's no-paste path was a docs refresh. The pastes turned it into the full execution path.

## 2. Implemented

- Docs truth (#465). The board, paste packet and handover now carry the live tip, the four pastes, and the fact that the 2026-09-17 "MERGEABLE + Julia green" status was void. All four DRAFTs branched from `7a6fe4962`, before the #423 rename.
- All four DRAFTs were rebased onto the renamed package, ported from `GLLVM` to `GLLVModels`, re-tested, and merged in the planned order:
  - #409 @ `137cab8e1`
  - #410 @ `decbc8ddc`
  - #399 @ `94a7b56f9`
  - #411 @ `3b19b2817`
- S4. Wiring fixes (#469 @ `194e01f0e`): the Julia home is passed as a directory, and a probe-only environment was added. The probe ran with a probe-only `GLLVM` shim, Shinichi's option A.
  - Result: `pass=0 fail=2 oracle_defect=2`. The frozen recorder's runner never attaches testthat, so both tests stop before any fit.
  - Receipt: #472.
  - Shinichi then chose (b): a corrected recorder commit was requested on gllvmTMB#1283.
- Track A on Totoro:
  - Pre-run passed at 09:42.
  - The full run took 56 min: 14 of 17 required cells succeeded; the 3 failures are the #323 holdouts.
  - Receipt: `docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md`.
- Delta A.
  - ACCEPTED (A) block recorded; the delta default is now `disp_group = :species`.
  - D1 remeasure: both cells PASS, with no tolerance changed.
    - lognormal: SE rel 4.0e-5; logLik Δ 1.8e-8, against −1.923 on 2026-09-15. The 2026-09-15 baseline's R library is unknown (see the `GLLVM_PARITY_R_LIBS` finding), so the PASS stands on its own; attributing the change to `disp_group` is inference.
    - Tier: each-own-optimum, one seed per cell.
    - gamma: SE rel 3.1e-5.
  - Receipts in #470.
- D3 Stage 1.
  - #411 now carries a confirmed fix for a pin-scaling bug that #411 itself introduced.
  - The Stage 1 slice (#471, merged `4d0569534`) adds the fit-time pin path and the confirmatory `loading_profile` export. It refuses predictor-informed, AGHQ, masked and offset fits, and adds one frozen-R NLL check at a fixed parameter point (Δ = 0.0). That check tests the packing convention and the Gaussian kernel only, not a constrained fit. Runbook item 3, an R-aligned pin-and-refit grid cell, is NOT met: the R reference has per-trait intercepts, which the Stage 1 `X = nothing` path cannot fit.
- Check-log sections from the four PRs were moved out of their PRs, so each merge stopped conflicting with the next. They land verbatim in this PR.

## 3a. Decisions and Rejected Alternatives

- Shinichi's decisions, in chat:
  - All four pastes. Delta A was accepted after a plain-language explanation that existing Julia delta fits will change.
  - "Yes, all four as DRAFT": pre-rebase now, rather than waiting for each paste.
  - D-280 head-to-head: deferred to the next clean plan.
  - "Claude runs it here" for Track A; the runbook had named Codex.
  - "go Track A".
  - S4: first "A" (the shim), then "b" (ask for a new recorder commit).
- Merge order #409 → #410 → #399 → #411. The two PRs that only add test scaffolding went first. #399 merged before its D1 remeasure, because the D1 runbook makes the merge a prerequisite.
- The pin-scaling fix was folded into #411 itself, so `main` never carried the bug. It was not left to the Stage 1 slice.
- Rejected:
  - Loosening any tolerance.
  - Editing the gllvmTMB recorder from this repo (D-220).
  - Treating the advisory Frozen R failure as blocking. It is `continue-on-error`; it failed the same way (278 pass, 8 fail, Julia 1.13.0) on every PR run in this lane, and on main's earlier run 35999539704 at `6ba1770ab`. Main had no completed CI run for today's merge heads.

## 4. Files Touched

- Merged PRs: #465, #409, #410, #469, #399, #411, #470, #471, #472.
- This PR:
  - `docs/dev-log/check-log.md` (carried sections plus this entry)
  - `docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md`
  - `docs/dev-log/core070/totoro-323-track-a-20260924/` (raw Track A evidence)
  - this report
  - `docs/dev-log/handover/2026-09-24-claude-handover-closeout.md`
  - the pending board, paste packet and `AGENTS.md` snapshot
- Outside the repo:
  - Totoro run folder `~/gllvmodels-track-a-20260924/`
  - scratch R library `~/local-scratch/R-gllvmtmb-frozen-b4d5fee64`
  - read-only recorder worktree `~/local-scratch/gllvmtmb-s4-recorder-97214679c`
  - a comment on itchyshin/gllvmTMB#1283
  - vault note D-280 (one deferral line)

## 5. Checks Run

- Acceptance ledger: `.unlazy/true-parity-20260924/`, re-verified with `gate-check --reverify` after every builder return.
  - B leaves: 6 of 6 each at merge time.
  - X leaves: see section 10 for the two honest abandons.
- Julia CI: 8 of 8 shards green on every merged head. Only the advisory Frozen R job was red, as on `main`.
- Local:
  - `test_second_order_delta_followup.jl`: 43 pass, 1 broken without R; with R, both D1 cells PASS.
  - `test_loading_profile_confirmatory.jl`: 48 pass, 1 broken.
  - Helper tests: S4 22/22, Track A 20/20, shard selection 43/43.
- Totoro: oracle `BUILD_EXIT=0`, `VERIFY_EXIT=0`; `runparity.jl` exit 1, from exactly the three holdouts.
- Completion panel (D-43), 2026-09-24: first pass NOT READY (Rose, 19 items; a recount found the 13-of-16 cell count wrong). Corrections were applied in `4faf3bb0f` and `1df7128e3`; the re-check returned READY WITH EDITS, and those edits are in this PR.
- Plan-vs-actual open items: 1 (X409 G2) accepted as disclosed drift; 2 and 3 resolved by the #471 and #472 merges and by the panel; 4 (X410 G2) turns met only when this PR merges the receipt; 5: no wording claims D1 gated the #399 merge; 6: each of the lane's test files is included exactly once in `test/runtests.jl` on main, with no duplicates; 7: surfaced in the handover Gotchas and the CHANGELOG.

## 6. Tests of the Tests

- Pin fix regression test: fails on the old line (−0.8 read back as −1.1765) and passes on the fix. An independent verifier reproduced both and matched an independent constrained MLE to 8 decimal places.
- Ledger checks:
  - The old-name check was shown to fail on #411 before its port (12 code lines) and to pass on #409 (prose only).
  - The include check was shown to fail on a stale branch and pass on `main`.
  - The include check was later changed to evaluate the merge result, after it measured the wrong thing (a stale branch file).
- D1 remeasure: the frozen gllvmTMB build was confirmed by path and `.so` hash, because the tool path ignores `GLLVM_PARITY_R_LIBS`.

## 7a. Issue Ledger

- #323: CLOSED 2026-09-15 as an agent-applied waiver (option B), which is not evidence of a pass. Today's Track A run is option (A) evidence, filed without reopening or commenting on the issue; the three holdout cells stay OWED.
- gllvmTMB#1283: recorder fix requested (S4 option b).
- The open items are in section 10.

## 8. Consistency Audit

- Foreign overlap was audited against every open PR.
  - The Stage 1 slice was moved off `src/confint_derived.jl` and `docs/src/derived-confidence-intervals.md`, which Codex PR #437 is editing.
  - Overlaps with #363 and #314 (retracted) and with #444 (the shared append-only check-log only) were judged harmless.
- The same bug class was checked elsewhere. Main's Stage 0 test helper does not carry the pin-scaling bug.
- Stale claims:
  - The handover's "#453 latte default ON" was corrected (#463 reverted it).
  - The handover's "#399 merge state UNKNOWN" was corrected: it was CONFLICTING.

## 9. What Did Not Go Smoothly

- All four DRAFTs predated the package rename, so their old green CI was void and each needed a port.
- Every PR also edited the shared `check-log.md`, so each merge re-conflicted the next. That chain was broken by carrying the sections.
- Two gates in my own ledger were wrong at first:
  - The include check looked at the stale branch instead of the merge result.
  - The Track A G2 check expected a gradient that only the full run can produce.

  Both were fixed before being relied on, with the reason recorded.
- Two false starts on the Totoro pre-run: a missing juliaup channel, and a full-history clone.
- `lane_launch.sh` overwrote `main`'s tracked `LOOP/` kit in its local scaffold commit. That was caught before any push, and fixed separately in the vault.
- S4's estimate line was written after the run, not before. The receipt says so.

## 10. Known Residuals

- S4: no endpoint numbers yet. Waiting for a new recorder commit on gllvmTMB#1283. Then re-run without the shim and remove the shim. The ledger's X409 G2 is abandoned (estimate not pre-run).
- Track A:
  - NATIVE-06 did not reach its R check (seeded-data guard on Julia 1.10.12, likely D-275, AGENT-INFERRED).
  - NATIVE-10 records no R gradient.
  - NATIVE-12's R gradient is still 5.90e-4 on Totoro, but it passes in the Julia 1.13.0 CI job, where NATIVE-10's Cell 9 fails instead. Holdout outcomes depend on the Julia version; none counts as a pass.
  - The ledger's X410 G4 is abandoned for that reason.
- Stage 1:
  - The ledger row `namespace/export/loading_profile` is untouched (no paired fixture evidence yet).
  - R's `maps.tsv` pins L11 = +0.8, while the Stage 0 fixture uses −0.8.
  - The heavy grid needs its own D-139 acknowledgement.
- Tooling: `tools/core070_second_order` ignores `GLLVM_PARITY_R_LIBS`. A run that sets only that variable silently uses the default gllvmTMB. It is not known which gllvmTMB the 2026-09-15 D1 FAIL receipts used.
- Delta A: one offset test reached a different optimum under the new per-trait default and is pinned to `:shared`. The new default can change convergence for some existing fits.
- Docs: `test_variational_dgamma.jl` fails when run alone, because it lacks `using LinearAlgebra`. This predates the lane.

## 11. Team Learning

- A DRAFT's green CI is evidence about its head only. A package rename underneath it voids that evidence, even without a textual conflict.
- A shared append-only log in every PR serializes the merges. Carrying sections to one closing PR removes the conflicts.
- **A gate must measure what lands (the merge result), not what a stale branch holds.**
- An environment variable a tool claims to honour must be proved loaded, by path and hash, before a numeric receipt counts.

## 12. Cross-Product Coverage

This work does NOT cover:
- any parity claim beyond the numbers in the three receipts;
- closing or waiving #323;
- promoting any capability row to "covered";
- a `Project.toml` bump, tag or release;
- any gllvmTMB change;
- the heavy Stage 1 grid;
- the S4 endpoints, which await the recorder fix.

The Julia CI green checks here cover the Julia test suite only. The frozen R smoke stays advisory.
