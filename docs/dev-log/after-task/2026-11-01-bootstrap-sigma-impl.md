## 1. Goal

Implement the pinned-R `bootstrap_Sigma` behaviour target (gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9, `R/bootstrap-sigma.R`, read-only) as a new, separate function `bootstrap_Sigma_r`, with red-first tests and a mutation record. v1 scope: `GllvmFit` supported; `TwoLevelFit` stretch; everything else refused with an explicit error.

## 2. Implemented

`src/bootstrap_sigma_r.jl` (new; 362 lines at first write, 403 after review) defines `BootstrapSigmaR` and `bootstrap_Sigma_r(fit; n_boot=999, level, what, conf=0.95, seed=nothing, n_cores=1, progress=true, keep_draws=false, link_residual=:auto, y, n_sites, X, Σ_phy, individual, center)`.

- One batch of `n_boot` datasets is drawn first from one stream (`MersenneTwister(seed)`, else the default RNG); the refit loop draws nothing. One refit per replicate; every summary comes from that refit.
- A replicate fails if the refit throws or returns a non-converged fit. Failed draws are NaN-shaped. A converged refit with a NaN entry is not failed; its entry is excluded.
- Bounds are type-7 percentiles (`_derived_percentile`) of the finite draws; no minimum-survivor rule. `n_effective` exists for vector summaries only (R limitation, kept). The attrition warning runs over vector entries only.
- `GllvmFit` (no W tier, no pins, no phylo block, no retained record): tier `B` only, taken from `_r_tier_total(fit, :unit)`. For `has_diag` with `K_W == 0` that is the identified total `sigma_y_site(fit)`, which is R's unit tier for that model. `ICC` is dropped with an `@info`; `cross_corr` is always skipped (R needs a multinomial trait).
- `TwoLevelFit` (pass `individual`): tiers `B`, `W`, and `ICC_site`.
- Refused with `ArgumentError`: `K_W > 0`, retained Gaussian record, `lambda_constraint` pins, phylo block or `Σ_phy`, any other fit class (including `PrecisionMultivariateFit`, `JointPhyloGroupedGaussianFit`, non-Gaussian), `n_cores > 1`, unknown level/what/link_residual, `conf` outside (0,1), `n_boot < 1`, all requested tiers absent, nothing reportable.
- Warnings never throw: `n_boot < min_boot` (39 at 0.95) gives the ceiling warning only; otherwise `n_boot < 999` gives the noise warning only.

`test/test_bootstrap_sigma_r.jl` (new, 570 lines): 17 testsets, 211 assertions (testsets 14-17 added after review; see section 5). `src/bootstrap_sigma_r.jl` is now 403 lines. The ledger now carries a "Gates draft", the "Spec", the deviations, and the mutation record.

## 3a. Decisions and Rejected Alternatives

- The Spec wins over the drafts. The code draft (n_boot 500, seed 0, long table, `n_used >= 10` rule, W and ICC refused) and the test draft built on it were not used; their contradictions are listed in the ledger.
- A new name, not a replacement: legacy `bootstrap_Sigma` is a different estimand and sits in a file deferred behind PR #1036.
- Rejected adding the 20-line helper extraction in `src/twolevel.jl` (outside the lease). The TwoLevelFit simulator loop is duplicated instead.
- Rejected "fixing" R's floating point or R's matrix-entry attrition gap; both are copied and documented.
- Added one refusal the Spec does not name (nothing reportable) to avoid running `n_boot` refits for an empty result.

## 4. Files Touched

- `src/bootstrap_sigma_r.jl` (new)
- `test/test_bootstrap_sigma_r.jl` (new)
- `docs/dev-log/check-log.d/2026-11-01-bootstrap-sigma-impl.md` (new)
- `docs/dev-log/after-task/2026-11-01-bootstrap-sigma-impl.md` (new)
- `docs/dev-log/plans/2026-11-01-bootstrap-sigma-checkpoint-ledger.md` (appended)

Nothing staged or committed. Not edited: `src/confint_derived_wald.jl`, `src/GLLVModels.jl`, `test/runtests.jl`, `test/test_se_machinery.jl`, `src/twolevel.jl`.

## 5. Checks Run

- Red-first: the same test file against a source copy without the driver gave 4 pass, 1 fail, 13 errored of 18.
- Green, final: 211 pass of 211 in the scratch project (testsets 14-17 added after review; one earlier test bug, a shadowed variable `n`, fixed by renaming it `nb41`; no assertion changed, no tolerance loosened). Not re-timed here.
- Green, first version: 154 pass of 154 (33.1 s in-test, 36.5 s wall). The first run had 2 failures from my own test arithmetic at conf = 0.9; the test was changed to state R's floating-point `min_boot = 20`; no source and no tolerance changed.
- Mutation harness: baseline green; 17 of 17 mutants killed (record in the ledger and `check-log.d`). Run on the 154-assertion version; NOT re-run on the 211-assertion file.
- Pre-run timing (D-287): n_boot = 10 on p = 4, n = 150 took 1.6 ms after compile; first call 2.3 s with compile.
- Julia was run from a scratch project outside the worktree (see section 9).

## 6. Tests of the Tests

Each mutant is a deliberate defect in a scratch copy of the driver, run against the unchanged test file: wrong tier (sigma_y_site, extract_Sigma unit_obs, TwoLevel W reading B), wrong summary (communality as a diagonal, ICC complement), wrong quantile (alpha not halved, swapped tails), repeated refit (10 per replicate), `seed + b`, ignored seed, failure counted by non-finite only, converged-NaN counted as failed, zero-filled failures, min_boot off by one, attrition threshold 0.5, `unit_obs` aliased to B, ICC kept with one tier. All 17 were killed; the baseline stayed green. The `seed + b` mutant was initially a gap: only a seed-shift assertion kills it (batch(8) must not be batch(7) shifted by one), which I added before the run. Each assertion that compares a contaminated value (sigma_eps^2) has a control showing the gap is not rounding (>0.1 on the diagonal).

## 7a. Issue Ledger

- OPEN: PR #1036 sequencing. `bootstrap_Sigma_r` is not included by `src/GLLVModels.jl`, not exported, and not in `test/runtests.jl`. The test file includes the source into the module only when the name is absent. Wiring and the test shard are deferred.
- OPEN: signature. The estimand signature evidence is Shinichi's chat words "Sign, wait for #1036" (2026-11-01); the plan packet header still reads PROPOSED / UNSIGNED. Lease was granted per `lane_lease.sh` output (lane `claude:bootstrap-sigma-p1`, 8 h, 2026-11-01); I did not re-check it and claim no lease here.
- OPEN: GATES are not signed. The G-number mapping is a reconstruction; the Spec is a candidate, not a signature. The two-fixture reading of Gate 1 needs sign-off. Three draft-versus-Spec disagreements (has_diag with K_W = 0, W/ICC handling, TwoLevel phy) were resolved in the Spec's favour and need confirmation.
- OPEN: `test/test_se_machinery.jl:210-235` still pins the legacy method (deliberately not touched). Gate G6.d (rewrite to the new contract) is blocked behind #1036.
- Unsupported in v1: record fits, pinned fits, phylo fits, `PrecisionMultivariateFit`, `JointPhyloGroupedGaussianFit`, non-Gaussian fits, `n_cores > 1`, `cross_corr` (always skipped), phy tier.
- Unverified: `has_diag` with `K_W == 0` (B tier = identified total `sigma_y_site(fit)`) being equivalent to R's unit tier rests on a Julia comment (`_r_unit_folds_diag`, `src/extractors.jl`, #701). It was never compared with R output.
- Resolved in favour of the Spec (G6.a contradiction): for GllvmFit the W tier and ICC are dropped with an `@info` (error only when nothing reportable remains), and `has_diag` with `K_W == 0` is supported as the identified total. The draft gate text said refuse; gates stay unsigned.
- Unverified: equivalence of the marginal draw to R's random-effect redraw is argued and covariance-tested, not compared with R output; the fixed-effect/intercept `X` path has no dedicated test; the real `PrecisionMultivariateFit` and `JointPhyloGroupedGaussianFit` classes were not constructed (a dummy type is refused); R's default deprecation warning for B/W aliases is not mirrored (decision).
- Default `n_boot = 999` and `progress = true` (one `@info` per replicate) mirror R and need maintainer agreement.

## 8. Consistency Audit

Core extractors `extract_Sigma(:unit_obs)` and `extract_ICC_site(::GllvmFit)` are never called by the driver (they add sigma_eps^2). Test 2 shows the tier B value differs from `sigma_y_site` by exactly `sigma_eps^2` on the diagonal and from the `unit_obs` total, by absolute values computed from `fit.pars`. The legacy table API is unchanged. The oracle checkout was only read.

## 9. What Did Not Go Smoothly

- `julialauncher --project=.` fails in this worktree (no `Manifest.toml`, `Optim` not instantiated). I used a scratch project whose `src` is a symlink to the worktree `src`; I did not run `Pkg.instantiate` or write a manifest in the worktree. A reviewer should re-run with a properly instantiated project.
- The code draft and its test draft followed an older design and had to be discarded.
- Two of my first test assertions were wrong (R's float min_boot at 0.9), which I only found by running.
- The Spec's TwoLevel helper extraction was out of lease, so one simulator loop is duplicated.

## 10. Known Residuals

- Not wired, not in CI shards, not exported; row status unchanged (REGISTRATION-ONLY, unbound).
- The duplicated TwoLevel simulator can drift from `repeatability_bootstrap_ci`.
- Blocked: PR #1036 hook-in. No full-suite run, no R comparison, no large-p timing (p = 80, 999 replicates is estimated at 37 to 44 h from recorded figures and needs a plan and approval under D-287; the Gaussian closed-form refit here is far cheaper, so that estimate does not transfer).
- Independent reviewer re-run (G6.e) not done; I am the author.

## 11. Team Learning

R's `ceiling(2/(1-conf)) - 1` is 20 at conf 0.9 because of floating point; copy R, test the actual number. Mutation testing found the `seed + b` hole in a draft-time test set; a single-stream design needs an explicit shifted-batch assertion. A worktree without a Manifest cannot run `--project=.`; a scratch project with a symlinked `src` is a clean workaround that leaves the tree untouched.

Memory receipt: ledger, Spec and drafts at the scratchpad path were the evidence; the repository source was read at the cited lines. The router manifest was not consulted.

Golden Set: not run; no existing implementation class was changed.

## 12. Cross-Product Coverage

This checkpoint covers only the GLLVModels.jl Gaussian `GllvmFit` and `TwoLevelFit` bootstrap route. It does NOT cover gllvmTMB (read-only, not modified), non-Gaussian families, the phylogenetic or precision classes, the legacy `bootstrap_Sigma`, or any other downstream project.
