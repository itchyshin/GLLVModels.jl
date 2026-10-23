# data/RD-FUNGI-BINOMIAL, second pass against four specific candidates

Date: 2026-10-21. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Read-only: no source, test, receipt, GATES.md, or existing note was changed; Julia was not started; nothing was committed. The tolerances stay 0.0001 (beta), 0.0001 (`Lambda Lambda'`), and 1e-06 (training linear predictor), as written in `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-FUNGI-BINOMIAL.json`.

Verdict: **NO_NEW_BUG**.

## Starting point

The first pass (`docs/dev-log/2026-10-20-p1-miss-fungi.md`, `docs/dev-log/2026-10-20-p1-overnight-fungi.md`) already found: log-likelihoods agree to `6.151276465971023e-08`; the logit weight equals the second derivative of the coded log-density; the covariate-mode and generic-Laplace objectives agree to 0; and the analytic gradient matches a central difference to `2.8218348546715788e-8`. Three quantities still miss: beta `0.000123932525228021` on `traitsp54`, `Lambda Lambda'` `0.00047443752671583184` on the `sp54` diagonal, and the training linear predictor `0.00032827901245191526` at site 133, trait `sp54`. This pass checks four different candidates the earlier passes did not: trial counts, covariate column order, species order, and which linear predictor each engine reports.

## Trial counts

`gllvm::fungi` is a presence or absence table: the receipt's data rule reads "species with prevalence in [.05,.60]" (`RD-FUNGI-BINOMIAL.json:36`), which only makes sense for a 0/1 response. R fits with `stats::binomial()` (`tools/true_parity/campaign/run_R.R:99`), the ordinary Bernoulli family with no trial-count argument. Julia's `fit_gllvm_cov` takes a trial-count matrix `N` that defaults to all ones (`src/families/covariates.jl:461`, read at `src/postfit.jl:3065`: `Nm = N === nothing ? fill(1, p, n) : N`). `run_J.jl` calls `gllvm(f, Y, covdata; family = fam, K = 2)` with no `N` argument (`tools/true_parity/campaign/run_J.jl:206`), so Julia also fits Bernoulli trials. Both sides use one trial per cell. This is not the defect.

## Covariate column order

R's design is `0 + trait + TEMPR + PRECIP + logAREA` (`run_R.R:98`), giving `beta_names` `traitsp01 .. traitsp59, TEMPR, PRECIP, logAREA` (confirmed in the raw receipt, `fungi_R.json.gz`). Julia's formula is `@formula(y ~ 1 + TEMPR + PRECIP + logAREA)` (`run_J.jl:203-204`), reaching `_build_site_modelmatrix` (`src/formula.jl:58-83`), which keeps the RHS terms in the order they appear in the formula (`non_const` at `src/formula.jl:61`, `modelmatrix` at `src/formula.jl:80` over that same order) before `gllvm` copies that matrix into `X[t, s, k] = mm[s, k]` (`src/formula.jl:337-339`). `write_receipts.py` asserts this at receipt-build time: it raises `SystemExit` unless R's name vector equals `["trait" + sp for sp in trait_levels] + covs` with `covs = ["TEMPR", "PRECIP", "logAREA"]` (`tools/true_parity/campaign/write_receipts.py:225-229`). The receipt built without that exit firing, so the three slope coefficients are matched by the same TEMPR/PRECIP/logAREA order on both sides; the first-pass note already recorded that all three slopes sit inside 0.0001 (largest gap `4.842831192e-06`, on `TEMPR`). Column order is not the defect.

## Species order

R builds the long frame with `sp <- setdiff(names(w), c(id, covs))` (`run_R.R:68`), which keeps the wide CSV's own column order (R's `setdiff` does not sort; it filters `names(w)` in place). Julia's `sp = filter(h -> !(h in covs) && h != "site", hdr)` (`run_J.jl:198`) does the same thing to the same header line. Decompressing the two committed raw files confirms the result is identical: R's `trait_levels` and `LLt_trait_names` both read `sp01, sp02, ..., sp59` in that order, and Julia's `trait_levels` in `fungi_J.toml.gz` reads the same `sp01, sp02, ..., sp59`. Species order is not the defect.

## Which linear predictor each engine reports

R stores `fit$report$eta` (`run_R.R:137`), TMB's own training-data linear predictor at the final parameter estimate, built from the long frame in which `trait` is the fast-moving index within each `site` (`long_from_wide`, `run_R.R:68-70`: `site` repeats each value `p` times, `trait` cycles through all `p` levels for each site). Julia stores `vec(predict(fit, Y, X; type = :link))` (`run_J.jl:223`), where `predict` is the three-argument `GllvmCovFit` method at `src/postfit.jl:3084-3092`: it recomputes the per-site latent mode through `getLV` (`src/postfit.jl:3062-3075`, the same offset-aware Laplace mode construction used throughout this fit) and returns `η = β + Xγ + Λẑ`. `vec` on a `p × n` matrix is column-major, so it also moves through trait fastest within each site column. Both vectors are 17700 long (`p = 59`, `n = 300`), in the same trait-fastest order, on the same `η = β + Xγ + Λz` quantity, at the same (converged) parameter point. Checking the one index the first-pass note names, flattened position 7900 (0-based: site `133`, trait `53` within the site, i.e. `sp54`), the quoted R and Julia values are the two R/Julia values already in the first-pass note, not a different pair pulled from a misaligned index. This is not the defect.

## Why this still does not change the verdict

All four candidates the earlier passes had not checked (trial counts, covariate column order, species order, which linear predictor) check out identical between the two engines on this cell. The three quantities that miss their bars (beta, `Lambda Lambda'`, linear predictor) each concentrate on the same species, `sp54`, and its two near-neighbours (`sp45`, `sp47`), which is the species R's own engine flags by name as a runaway loading (`RD-FUNGI-BINOMIAL.json:59-60`: prevalence 0.0433, max loading 2.62). R's own gradient at its returned point is `0.00386035726088885`; Julia's is `7.5812079072257177e-06`. On a species this close to quasi-separation, a shallow likelihood direction lets two different small gradients at convergence correspond to parameter values a few ten-thousandths apart while leaving the log-likelihood itself within `6.15e-08`, exactly the pattern the three failing cases show. No new defect was found in this pass.

`data/RD-FUNGI-BINOMIAL` is not admitted.

NO_NEW_BUG
