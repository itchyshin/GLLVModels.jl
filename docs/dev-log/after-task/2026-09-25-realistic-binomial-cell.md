# After-task: realistic-size Binomial-logit T4 cell (2026-09-25)

Status: COMPLETE, 1/1 PASS. Updated 2026-09-26 following independent review of PR #491
(`reviews/pr-491.md`): removed "bug" framing for the intercept change, added the discarded
first run as a labelled separated-regime record, added a 5-seed panel, corrected the CI-
endpoint row to a formal PASS, withdrew the A13 binding proposal and restated it toward A7,
and recorded script SHA-256s in the receipt. See
`docs/dev-log/after-task/2026-09-26-pr491-review-followup.md` for the follow-up after-task.

Branch: `claude/realistic-binomial-cell-20260925`. Local worktree:
`~/local-scratch/gllvm-a05-binomial`. Totoro worktree: `~/gllvmodels-a05-20260925`, left in
place, not modified after this run. Existing Totoro checkout, untouched:
`~/gllvmodels-track-a-20260924/GLLVModels.jl`.

This run fills the Binomial-logit gap in the T4 p6 realistic-size grid
(`docs/dev-log/after-task/2026-09-05-totoro-t4-p6-grid.md`, PR #297).

Claim boundary: this is one realistic-size (RSZ) cell. It is second-order tolerance
evidence for the Binomial-logit family at p=20, n=500, K=2. It is not a true-parity claim,
and it does not by itself promote any gate-tier row.

## Scope

Family: Binomial-logit (Bernoulli trials, `N=1`), p=20, n=500, K=2, seed=42. Host: Totoro
(`snakagaw@totoro.biology.ualberta.ca`), reached through the existing SSH ControlMaster
socket, no fresh login. Julia 1.10.12 (juliaup), 4 threads requested
(`JULIA_NUM_THREADS=4`), `OPENBLAS_NUM_THREADS=1`, one job at a time on a shared machine
(384 cores; other lanes' unrelated jobs were running concurrently and were left alone). R
library: the frozen `gllvmTMB` 0.7.0 build at
`~/gllvmodels-track-a-20260924/GLLVModels.jl/.unlazy/r-build/library`, pinned to commit
`b4d5fee64def88bc768dda1f1f77c29b295edd86` (checked against the build's own
`CORE070_SOURCE_PIN.toml`).

Why this cell: the existing T4 p6 receipts
(`docs/dev-log/core070/t4-p6-*-receipt-2026-09-05.json`, from PR #297) cover Gaussian,
Poisson and NB2 at realistic size, but not Binomial. The gate-tier doc
(`docs/dev-log/core070/true-parity-gate-tier-2026-09-05.md`) lists row A13
(`covariance/COV-ORD-LATENT-BARE-RSZ`, ordinary `latent()` bare ordination at RSZ tier) as
"Realistic-size owed". The second-order parity contract
(`docs/dev-log/core070/second-order-parity-contract.md`, section 6) names Binomial-logit as
one of the five signed batch-1 families (Gaussian, Poisson-log, Binomial-logit, Beta-logit,
NB2-log): it uses a canonical link, so Fisher and observed curvature agree exactly, the
same as Poisson. Beta-logit was deliberately excluded from this run; it waits on PR #483,
per instruction.

## Scripts

Used the existing T4 scripts without changing their mechanism
(`tools/core070_realistic_size_cell.jl`, `tools/core070_realistic_size_cell.R`,
`tools/t4_p6_write_receipt.py`), and added the one case they lacked: `binomial`.

- Julia side (`tools/core070_realistic_size_cell.jl`): new `elseif fam == "binomial"`
  branch, structured like the existing `poisson` branch, using `fit_binomial_gllvm`,
  `confint`, and `_family_ci`/`_fd_hessian` for `cond(H)`. One deliberate change from the
  shared data-generating setup: the file's shared intercept
  (`β_log = log.(2.0 .+ 3.0 .* rand(p))`) is tuned for count-family rate scale and gives
  logit intercepts 0.69 to 1.61, i.e. true trait prevalence 0.66 to 0.83 with latent SD
  about 0.5 — an ordinary range for real presence/absence data, not a malformed DGP. The
  first run, using it as-is, landed in the quasi-separation regime: Julia reported
  `converged=false`, and the frozen R oracle converged (`convergence=0`) with a "runaway
  trait loading" warning (trait t16, prevalence 0.814, `max_loading=44.3`,
  `relative_loading=120`); the two engines' logLik agree to 0.02. This is an ordinary DGP
  that landed in a hard regime by chance; it is a real, reproduced parity observation, kept
  as a labelled record rather than discarded (see "First run" below and
  `docs/dev-log/core070/binomial-p20-n500-K2-separated-regime-seed42.md`). The binomial
  branch instead draws its own zero-centered logit intercept
  (`β_bin = 0.4 .* randn(p)`), still a deterministic function of `(p, K, seed)`, reusing the
  same `Λ_true`/`Z` latent draw. The reason for a zero-centered intercept, stated on its own
  terms rather than as a correction: second-order parity (SE, vcov, Wald CI) needs an
  interior MLE, away from any boundary; `N(0, 0.4²)` logit intercepts keep expected
  prevalence near 0.34 to 0.65 (measured on seed 42: observed range 0.344-0.666), which is
  centered and away from the separation boundary this DGP can otherwise reach. A 5-seed
  panel under each DGP (excluding seed 42) confirms this is not seed-42-specific tuning: see
  `docs/dev-log/core070/binomial-p20-n500-K2-seed-panel.md`.
- R side (`tools/core070_realistic_size_cell.R`): added `binomial = stats::binomial()` to
  the family switch. Nothing else changed; the existing
  `value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE)` formula and
  `gllvmTMBcontrol(n_init = 1L, se = TRUE)` control already work for a 0/1 response.

`tools/t4_p6_write_receipt.py`, `tools/t4_p6_cells.tsv`, and both grid launcher/poll scripts
were not changed. This was a single ad hoc cell, run directly through the two per-cell
scripts, not a re-run of the 12-cell grid.

## Run

First run (count-scale intercept, discarded, not used for the receipt): `converged=false`, R
warned of a runaway loading on trait t16 (prevalence 0.814, `max_loading=44.3`,
`relative_loading=120`), quasi-complete separation from the DGP described above. Wall 185s
(Julia) plus 8s (R), 203s total. This is a real parity observation in the quasi-separation
regime, and it was discarded (not fixed) in favor of the zero-centered intercept because the
task needs an interior-MLE cell, not because the run was wrong. That
first run's log was not kept on Totoro, but it reproduces deterministically (seed 42, same
`Λ_true`/`β_log`/`Z` RNG draws): re-run 2026-09-26, ~2 minutes per engine locally against the
frozen R library, bit-identical prevalence (0.814) and trait (t16) to the numbers above; see
`docs/dev-log/core070/binomial-p20-n500-K2-separated-regime-seed42.md` for the full,
labelled record (Julia logLik -5292.804134629608, R logLik -5292.7859682457, engines agree
to 0.02).

Second run (used for the receipt): both engines converged cleanly.

- Julia: `converged=true`, `pd_hessian=true`, fit 11.07s, confint 47.56s.
- R: `converged=TRUE`, `has_sd_report=TRUE`, fit 4.96s; confint is already available from
  the fit's own `sdreport()`, so its separately timed portion is near zero.
- Total wall, this run, launch to finish: 113s. Well under the 10-minute estimate stated
  before this task started (comparable p20/n500 T4 cells ran 19 to 217s), and far under the
  30-minute stop condition.

Launch log for the receipt run: `docs/dev-log/core070/t4-p6-binomial-p20-n500-K2-launch-2026-09-25.log`.
The first run's own Totoro log was not kept; the reproduction above (local, 2026-09-26)
stands in its place and matches it exactly on every reported number.

A 5-seed panel under each DGP (seeds 1-5, excluding the receipt seed 42), run locally
2026-09-26 (Julia 181s + R 59s = 240s total, against a 15-minute estimate and the 30-minute
stop condition): the count-scale intercept separates on 2 of 6 seeds tried (42 and 5); the
zero-centered intercept separates on 0 of 6. Full table:
`docs/dev-log/core070/binomial-p20-n500-K2-seed-panel.md`. This shows seed 42 was not picked
because it happens to pass under the zero-centered DGP — none of the 6 seeds tried under
that DGP separate.

## Result

PASS, each-own-optimum tier, against the signed tolerances
(`second-order-parity-contract.md`, section 4). `r_condition_number` = 857.89, below 1e3, so
no conditioning scale-up applies; the base each-own-optimum bounds are relative ≤ 1e-2 for
SE and for the vcov Frobenius relative delta.

| Quantity | Julia | R | Delta | Tolerance | Pass/fail |
|---|---:|---:|---:|---|---|
| logLik | -6744.539486896926 | -6744.5394869632 | 6.63e-08 | not gated at this tier, recorded | n/a |
| SE, max relative delta (beta) | n/a | n/a | 8.08e-06 | relative ≤ 1e-2 | PASS |
| vcov Frobenius relative delta | n/a | n/a | 1.18e-05 | ≤ 1e-2 | PASS |
| Wald CI endpoint, max abs delta | n/a | n/a | 7.28e-06 | ratio to half-width ≤ 5e-2 | PASS (formal: max ratio 4.1e-05, computed from the committed CSVs, `binomial_p20_n500_K2_julia_terms.csv` vs `..._r_fixed.csv`, worst case term `beta[11]`) |
| Julia converged | true | n/a | n/a | must be true | PASS |
| R converged | n/a | TRUE | n/a | must be true | PASS |
| pd_hessian (native) | true | R has no equivalent flag; `has_sd_report=TRUE` | n/a | must be true | PASS |

`native_condition_number` (Julia) = 162.07. `r_condition_number` (R,
`kappa(solve(cov.fixed))`) = 857.89. Both are recorded per the contract; conditioning is not
itself gated at this tier, it only scales the SE and vcov tolerances when
`r_condition_number > 1e3`, which is not the case here.

Receipt: `docs/dev-log/core070/t4-p6-binomial-p20-n500-K2-receipt-2026-09-25.json` (plus
`.md`). Raw per-cell outputs (summary, terms, vcov CSVs, both engines):
`docs/dev-log/core070/t4-p6-out/binomial_p20_n500_K2_*`.

## What this does and does not establish

Does: gives Binomial-logit an each-own-optimum RSZ receipt at p=20, n=500, K=2, matching
the shape and tier of the existing Gaussian, Poisson, and NB2 T4 receipts, with all three
second-order quantities (SE, vcov, Wald CI) inside the signed each-own-optimum tolerances
and both condition numbers recorded.

Does not establish true parity (one seed, one shape, each-own-optimum, not
matched-coordinates); does not establish RSZ evidence at p=50 or n=2000 for Binomial (only
the T4 p6 gaussian/poisson/nb2 grid covers those cells); says nothing about Beta-logit
(excluded per instruction, blocked on PR #483); and does not resolve gate row A13 on its
own (see the proposal below).

## Gate row A13: proposal withdrawn (2026-09-26)

The original after-task proposed binding A13 (`covariance/COV-ORD-LATENT-BARE-RSZ`) to this
receipt plus the twelve existing K=2 T4-p6 receipts. Withdrawn: A13's route is BRG/NAT
(`true-parity-gate-tier-2026-09-05.md:49`) and is a **covariance** row (Λ Λᵀ). Every T4-p6
receipt, including this one, is native-only (`fit_*_gllvm`, not the formula bridge) and
compares only logLik and the β-block SE/vcov — never the covariance estimand A13 names. The
proposal also miscounted "five family/shape combinations" against four families (13
receipts). None of that is evidence toward A13 as written; `docs/dev-log/core070/true-parity-gate-tier-2026-09-05.md`
is SIGNED and this session did not edit it.

Restated instead as Binomial RSZ evidence toward **A7** (`BINOMIAL-LOGIT-2SO`): this receipt
adds an each-own-optimum RSZ point (p=20, n=500, K=2) to A7's second-order Binomial-logit
evidence, on the same terms as the Gaussian/Poisson/NB2 T4-p6 receipts contribute to their
own rows. A13 stays owed until a bridge route (BRG/NAT) with an actual covariance comparison
exists; this is a maintainer decision, not made here.

## Compute and cleanup

Two Julia and R runs on Totoro: about 113s wall for the clean run, plus about 203s for the
diagnosed and discarded first attempt, roughly 5 minutes of Totoro compute in total, at 4
Julia threads, `OPENBLAS_NUM_THREADS=1`, one job at a time. No process was left running on
Totoro after either run finished on its own (checked directly: no
`gllvmodels-a05-20260925` processes present after completion). The Totoro worktree
`~/gllvmodels-a05-20260925` is left in place per instruction, at `origin/main` commit
`d9bc77412`, with the two scripts overwritten by rsync (not committed there; Totoro is
scratch compute, not a git remote for this branch). `gllvmTMB`/`gllvmTMB.git` were not
touched.

## Reviewers

- Maintainer: on merging this draft PR. The A13 binding proposal above was withdrawn
  2026-09-26 and restated toward A7; no maintainer sign-off is being requested for A13.
- The engine flag-disagreement under separation (R converges with a warning; Julia reports
  `converged=false` with no diagnostic) is tracked in #498, alongside #494's two similar
  Binomial covariate fits; not fixed here.
