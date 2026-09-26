# After-task: PR #491 review follow-up (2026-09-26)

Status: COMPLETE. Applies the independent reviewer's items from `reviews/pr-491.md` (Gauss,
2026-09-26) to PR #491 (realistic-size Binomial-logit T4 cell). Branch:
`claude/realistic-binomial-cell-20260925`, worktree `~/local-scratch/gllvm-a05-binomial`,
rebased onto `origin/main` @ `d89179d41` (post #481/#483/#487/#488/#492/#494/#495) before
this work.

## 1. Goal

Fix the review's one BLOCKING item and its four SHOULD-FIX items without touching gllvmTMB,
the frozen-contract docs, or any required-cell assertion; land an honest record of the
discarded first run instead of calling it a "bug".

## 2. Implemented

1. **Removed "bug" framing everywhere** (PR body, `docs/dev-log/after-task/2026-09-25-realistic-binomial-cell.md`).
   States plainly: the first run used the original DGP (`Λ_true = 0.35*randn`, count-scale
   logit intercepts 0.69-1.61), landed in the quasi-separation regime by chance (an ordinary
   DGP, not a malformed one), Julia reported `converged=false`, R converged with a runaway-
   loading warning, and the two engines agree on logLik to 0.02.
2. **Committed the first run as a labelled separated-regime record.** Reproduced
   deterministically (seed 42, same `Λ_true`/`β_log`/`Z` RNG order as the tracked script; not
   itself a change to `tools/core070_realistic_size_cell.jl`), local Julia 1.10.12 + the
   frozen R library, 2026-09-26: Julia logLik -5292.804134629608 (bit-identical to the
   after-task's original number), R logLik -5292.7859682457, same trait (t16), same
   prevalence (0.814), same warning text, ~13s Julia + ~9s R wall (well under the reviewer's
   "~2 minutes per engine" estimate). Record:
   `docs/dev-log/core070/binomial-p20-n500-K2-separated-regime-seed42.md`.
3. **Justified `β_bin = 0.4*randn` on its own terms**: second-order parity (SE, vcov, Wald
   CI) needs an interior MLE; this intercept keeps expected prevalence near 0.34-0.65
   (measured on seed 42: 0.344-0.666), away from the separation boundary. Not framed as a
   correction to a mistake.
4. **5-seed panel under each DGP** (seeds 1-5, seed 42 excluded and reported separately),
   local, 2026-09-26. Estimated before running: ~1 min Julia + 15s R per fit, under 15
   minutes total. Measured: 181s Julia + 59s R = 240s, inside the 30-minute stop condition.
   Result: count-scale intercept separates 2 of 6 seeds tried (42, 5); zero-centered
   intercept separates 0 of 6. Shows seed 42 was not picked for passing under the receipt
   DGP. Full table: `docs/dev-log/core070/binomial-p20-n500-K2-seed-panel.md`.
5. **CI-endpoint row corrected from "PASS (informal)" to a formal PASS.** Recomputed from
   the committed CSVs (`docs/dev-log/core070/t4-p6-out/binomial_p20_n500_K2_julia_terms.csv`
   vs `..._r_fixed.csv`): max endpoint delta 7.279991290520815e-06 (unchanged from the
   receipt), max ratio to half-width 4.1014944464272155e-05 (worst case `beta[11]`), against
   the contract's 5e-2 bar. Updated in the after-task table.
6. **A13 binding proposal withdrawn**, restated as Binomial RSZ evidence toward A7
   (`family/BINOMIAL-LOGIT-2SO`, route NAT) instead: A13
   (`covariance/COV-ORD-LATENT-BARE-RSZ`) is a covariance row on route BRG/NAT, and every
   T4-p6 receipt (including this one) is native-only and compares logLik plus the β-block
   SE/vcov, never the covariance estimand (Λ Λᵀ) A13 names. The gate-tier doc
   (`true-parity-gate-tier-2026-09-05.md`) is SIGNED and was not edited.
7. **Script SHA-256 recorded in the receipt**
   (`docs/dev-log/core070/t4-p6-binomial-p20-n500-K2-receipt-2026-09-25.json`): added
   `script_sha256_jl`, `script_sha256_r`, `pr_head_commit`, and a `git_head_note` explaining
   why the recorded `git_head` (`d9bc77412`) does not itself contain the binomial branch
   (Totoro worktree updated by rsync, not committed there).
8. **Linked follow-up issue #498** (already open, opened independently for #494's flag
   disagreement) from the after-task and the separated-regime record: both engines land on
   the same degenerate solution under separation but flag it differently (R converges with a
   warning; Julia reports `converged=false` with no diagnostic). Not fixed here.

## 3. Verification

- Reproduced the receipt DGP (seed 42, `β_bin`) locally: Julia logLik -6744.539486896926
  (bit-identical to the committed receipt), R logLik -6744.5394869632 (bit-identical). Both
  engines converged, no warning.
- Recomputed the CI-endpoint ratio directly from the committed CSVs (script in
  `~/local-scratch/review-491-fix-scratch/`, not committed): confirms the reviewer's
  4.1e-05 exactly.
- Verified A7/A13 routes against `true-parity-gate-tier-2026-09-05.md` lines 43 and 49.
- `python3 ~/shinichi-brain/tools/agent_mention_check.py --text <PR body file>`: clean (no
  GitHub `@handles` for agents).

## 4. Compute

All local (no Totoro/DRAC): Julia 1.10.12 via juliaup (`julia +1.10`), `JULIA_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=1`; R via the frozen `gllvmTMB` 0.7.0 library
(`~/local-scratch/R-gllvmtmb-frozen-b4d5fee64`, `R_LIBS` set for the duration of each
`Rscript` call, not persisted). Total wall for all reproduction + panel runs: about 5.4
minutes (84s reproduction + 181s Julia panel + 59s R panel), against a stated pre-run
estimate of under 30 minutes.

## 5. Scope not covered

- Does not add the runaway-loading diagnostic to Julia, does not R-check #494's seeds 1/8,
  and does not decide whether Julia's `converged` flag should match R's under separation —
  all three are #498's scope, not this PR's.
- Does not touch Beta-logit, does not touch gllvmTMB, does not change the frozen-contract or
  gate-tier docs, does not mark this PR ready.

## 6. Reviewers

- Maintainer: merge decision for #491 (not requested here to be marked ready). No A13
  sign-off is being requested; the proposal was withdrawn.
