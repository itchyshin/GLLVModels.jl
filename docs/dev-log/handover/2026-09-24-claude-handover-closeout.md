# Session Handoff: true parity, four paste gates executed (closeout)

Meta: 2026-09-24 · from Claude (lane `true-parity-20260924`, Opus 5.5) · to Claude · supersedes the OWED list of `2026-09-24-claude-handover.md` (Cursor's pickup handover of this morning, kept as history).

You are Claude, picking up the GLLVModels.jl ↔ gllvmTMB true-parity programme. Chat history is not authoritative; reconcile this file with live `git` first.

## Critical Context

1. **All four paste gates were given and executed on 2026-09-24.** The four DRAFTs (#409 S4, #410 Track A, #399 Delta A, #411 Stage 1) are merged. Each gate was carried to its runbook stop line. Two produced numbers; two ended at a documented blocker (details below).
2. **gllvmTMB is read-only from this repo (D-220).** The S4 recorder fix was requested on itchyshin/gllvmTMB#1283; wait for its owning lane to answer.
3. **Multi-lane repo.** Foreign lanes are still open, and their files are off-limits:
   - Codex docs #433/#437/#439 (#437 edits `src/confint_derived.jl` and `docs/src/derived-confidence-intervals.md`)
   - Claude reader #444
   - the Cursor lease on `src/grouped_nongaussian_fit.jl`
   - tracked `.unlazy/` ledgers from other lanes (`grouped-analytic-20260920`, `s9c-coverage-448-20260922`, `temporal-ar1`, `totoro-t4-p6-grid`), which make `handoff_gate.sh` and the after-task checker fail; they are PROTECTED, not this programme's.
4. `Project.toml` stays `0.3.0`. The programme goal is not complete. Do not mark `/goal` done.

## What Was Accomplished

| Gate | Merged | Outcome |
|---|---|---|
| Docs truth | #465 `3b2fd96e3` | Board, paste packet and handover now carry the live tip and the four pastes. |
| S4 probe (#409) | #409 `137cab8e1`, wiring #469 `194e01f0e`, receipt #472 | Option A (probe-only `GLLVM` shim) ran. Result `pass=0 fail=2 oracle_defect=2`: the frozen recorder's runner never attaches testthat, so both tests stop before any fit. Shinichi then chose option (b), and a recorder fix was requested on gllvmTMB#1283. |
| Track A (#410) | #410 `decbc8ddc` | Totoro, 56 min, `main` `94a7b56f9`, oracle `b4d5fee64` built and verified. 13 of 16 required cells succeeded. See the per-holdout rows below. Receipt: `docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md`; raw evidence in `docs/dev-log/core070/totoro-323-track-a-20260924/`. |
| Delta A (#399) | #399 `94a7b56f9`, D1 receipts #470 `3c56e629c` | Delta default is now `disp_group = :species`. **D1 PASS on both cells**, no tolerance changed: lognormal SE rel 4.0e-5 (logLik Δ went from −1.923 to 1.8e-8), gamma SE rel 3.1e-5. |
| Stage 1 (#411) | #411 `3b19b2817`; slice #471 (see Landing State) | #411 includes a verified fix for a σ_eps pin-scaling bug it had introduced (pins were off by 1/σ_eps). The slice adds the fit-time pin path, the confirmatory `loading_profile` export, four admission refusals, and one exact frozen-R-oracle cell. |

Track A holdouts:
- NATIVE-12 still fails on the R side: `r_gradient_max` 5.90e-4.
- NATIVE-06 never reached its R check: a seeded-data hash guard refused on Julia 1.10.12 (AGENT-INFERRED as the D-275 class).
- NATIVE-10's parity cell passes (Δ logLik 2.0e-8); only its near-Gaussian diagnostic fails to converge.

Full report: `docs/dev-log/after-task/2026-09-24-true-parity-four-gates.md`.

## Current Working State

- Working: all merged items above are on `main`.
- Not finished: the S4 endpoints (waiting on gllvmTMB#1283), and the Track A holdout follow-ups.
- Nothing is running on Totoro from this lane (run folder `~/gllvmodels-track-a-20260924/` kept as evidence).

## Key Decisions & Rationale

Shinichi, in chat, 2026-09-24:
- the four pastes;
- "Yes, all four as DRAFT" (pre-rebase all four);
- D-280 deferred;
- "Claude runs it here" for Track A;
- "go Track A";
- S4: "A", then "b".

Lane decisions:
- Merge order #409 → #410 → #399 → #411.
- Check-log sections were moved out of each PR to stop merge-chain conflicts. They land in the closeout PR.
- The pin fix was folded into #411, so `main` never carried the bug.
- The advisory Frozen R CI failure was treated as non-blocking: it is `continue-on-error` and fails identically on `main`.

## Landing State

`handoff_gate.sh` (run on the closeout worktree): GATE FAIL on six foreign `.unlazy` ledgers and on many foreign unpushed local branches. Both are PROTECTED foreign work, as in this morning's handover. This lane's own ledger lives, git-ignored, in the lane worktree.

| Artifact | Committed | Pushed | PR | State |
|---|---|---|---|---|
| #465, #409, #410, #469, #399, #411, #470, #472 | y | y | merged | LANDED |
| Stage 1 slice `claude/d3-stage1-slice-20260924` | y | y | #471 | see the closeout PR comment for its final state |
| S4 receipt `claude/s4-probe-run-20260924` | y | y | #472 | LANDED (merged `aed31c8bb`) |
| This closeout (`claude/true-parity-closeout-20260924`) | y | y | this PR | LANDING |
| Lane ledger `.unlazy/true-parity-20260924/` in `~/local-scratch/lanes/GLLVM.jl-true-parity-20260924` | n/a (ignored) | n/a | none | run state; two gates honestly ABANDONED (X409 G2, X410 G4) |
| Foreign `.unlazy` ledgers and unpushed local branches | mixed | no | none | PROTECTED FOREIGN |

FINDINGS-OF-RECORD:
- The σ_eps pin-scaling bug (#411, commit `2acd8d27b`), confirmed by an independent verifier and fixed before merge.
- `tools/core070_second_order` ignores `GLLVM_PARITY_R_LIBS`. A run that sets only that variable silently uses the default gllvmTMB. It is unknown which gllvmTMB the 2026-09-15 D1 FAIL receipts used.
- Stage 0 fixture sign: R's `maps.tsv` pins L11 = +0.8, while this repo's Stage 0 fixture uses −0.8.

## Next Immediate Steps

1. OWED (waits on another lane): when gllvmTMB#1283 has a new recorder commit that attaches testthat and says `using GLLVModels`:
   - re-run the S4 probe without the shim (runbook `docs/dev-log/plans/2026-09-16-s4-probe-julia-checklist-paste-gated.md`);
   - file a new receipt;
   - remove `tools/destination_b/probe_env/GLLVM/`;
   - then examine the `phylo_covariance` 5e-6 gate. The diagnostic suggested a 2.2e-6 gap, but that run is not evidence.
2. OWED (decide with Shinichi): how the NB2 fixture (NATIVE-06) should be pinned across Julia versions; whether the Student cell should record `r_gradient_max`.
3. OWED (small, reversible): make `tools/core070_second_order/common.jl` honour `GLLVM_PARITY_R_LIBS` or fail closed, and add a test.
4. OWED (decide with Shinichi): the Stage 0 fixture's L11 sign against R's reference.
5. Not OWED without a new acknowledgement: the Stage 1 heavy grid on Totoro (D-139); ledger-row promotion; a `Project.toml` bump.

## Blockers / Open Questions

- S4 endpoints are blocked on gllvmTMB#1283 (another lane).
- The NB2 fixture pin and the Stage 0 sign need Shinichi.

## Gotchas & Failed Approaches

- A DRAFT's green CI describes its own head only. The #423 rename made the 2026-09-17 greens void.
- Every PR here used to add a section at the top of `docs/dev-log/check-log.md`, so each merge conflicted the next. Keep per-PR notes in the after-task file and land check-log entries in one closing PR.
- On Totoro:
  - use `julia +1.10.12` (no `1.10` channel);
  - fetch the frozen gllvmTMB pin with `--depth 1` (a full-history clone took 15+ minutes);
  - point `R_LIBS_USER` inside the run folder so `install_deps` never writes the shared library.
- The S4 recorder attests the shim's directory as the project path while the shim is in use; the receipt records this.

## How to Resume

```sh
cd "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"   # read-only here; work in a worktree
~/shinichi-brain/tools/lane_preflight.sh .
git fetch origin && git log --oneline -8 origin/main
gh pr list -R itchyshin/GLLVModels.jl --state open
gh pr view 1283 -R itchyshin/gllvmTMB --json headRefOid,updatedAt,comments --jq '.headRefOid, .updatedAt, (.comments | length)'
```

Frozen oracle R library (Mac): `~/local-scratch/R-gllvmtmb-frozen-b4d5fee64`. Load it through `R_LIBS_USER`, not only `GLLVM_PARITY_R_LIBS`.

```text
Read AGENTS.md and docs/dev-log/handover/2026-09-24-claude-handover-closeout.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```
