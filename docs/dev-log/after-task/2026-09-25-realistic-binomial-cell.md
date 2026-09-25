# After-task: realistic-size Binomial-logit T4 cell (2026-09-25)

Status: COMPLETE, 1/1 PASS.

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
  (`β_log = log.(2.0 .+ 3.0 .* rand(p))`) is tuned for count-family rate scale (2 to 5) and
  is not a sensible logit-scale intercept; it puts baseline prevalence at 0.66 to 0.83
  rather than centered near 0.5. A first attempt using it as-is drove one trait into
  quasi-complete separation (`converged=false`; R warned of a "runaway trait loading",
  prevalence 0.814, loading ratio 120x). The binomial branch now draws its own
  zero-centered logit intercept (`β_bin = 0.4 .* randn(p)`), still a deterministic function
  of `(p, K, seed)`, reusing the same `Λ_true`/`Z` latent draw.
- R side (`tools/core070_realistic_size_cell.R`): added `binomial = stats::binomial()` to
  the family switch. Nothing else changed; the existing
  `value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE)` formula and
  `gllvmTMBcontrol(n_init = 1L, se = TRUE)` control already work for a 0/1 response.

`tools/t4_p6_write_receipt.py`, `tools/t4_p6_cells.tsv`, and both grid launcher/poll scripts
were not changed. This was a single ad hoc cell, run directly through the two per-cell
scripts, not a re-run of the 12-cell grid.

## Run

First attempt (bad data-generating setup, not used for the receipt): `converged=false`, R
warned of a runaway loading on trait t16 (prevalence 0.814, `max_loading=44.3`,
`relative_loading=120`), quasi-complete separation from the mis-centered intercept
described above. Wall 185s (Julia) plus 8s (R), 203s total. Diagnosed as a bug in this run's
setup, not a Julia or R engine defect, and fixed before any receipt was written.

Second attempt (used for the receipt): both engines converged cleanly.

- Julia: `converged=true`, `pd_hessian=true`, fit 11.07s, confint 47.56s.
- R: `converged=TRUE`, `has_sd_report=TRUE`, fit 4.96s; confint is already available from
  the fit's own `sdreport()`, so its separately timed portion is near zero.
- Total wall, this run, launch to finish: 113s. Well under the 10-minute estimate stated
  before this task started (comparable p20/n500 T4 cells ran 19 to 217s), and far under the
  30-minute stop condition.

Launch log for the successful run: `docs/dev-log/core070/t4-p6-binomial-p20-n500-K2-launch-2026-09-25.log`.
The first, failed-setup run's log was not kept; it reflects a bug in this session's DGP that
was fixed before any receipt existed.

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
| Wald CI endpoint, max abs delta | n/a | n/a | 7.28e-06 | ≤ 5e-2 of interval half-width; not separately computed here, but this delta sits five to six orders below the interval scale itself | PASS (informal, given the SE and vcov margins) |
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

## Proposal for gate row A13 (not applied; the gate-tier doc is signed and was left alone)

`docs/dev-log/core070/true-parity-gate-tier-2026-09-05.md` is SIGNED, and this session did
not edit it. For the record, and for whoever next reviews A13
(`covariance/COV-ORD-LATENT-BARE-RSZ`, "Realistic-size owed"): the ordinary
`latent(0 + trait | site, d = K, unique = FALSE)` mechanism this row is about is exactly
what every T4 p6 cell (Gaussian x4, Poisson x4, NB2 x4) and this Binomial cell all fit.
Five family/shape combinations now carry an each-own-optimum RSZ receipt with that same
latent term at K=2: Gaussian at p{20,50}/n{500,2000}, Poisson on the same grid, NB2 on the
same grid, and now Binomial-logit at p=20/n=500.

Proposal: bind A13 to those twelve K=2 T4-p6 receipts plus this Binomial-logit receipt, 13
in total, with the same caveats the T4-p6 after-task already states: RSZ scaling and
tolerance evidence, not a true-parity or matched-coordinates claim, and no p=50/n=2000
Binomial cell yet. This is a proposal for maintainer sign-off. It is not a change to the
gate-tier doc.

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

- Maintainer: sign-off on the A13 binding proposal above, and on merging this draft PR.
