# Review: PR #563 — temporal slice 2 (unit / unit_obs composition at gllvmTMB P1)

Reviewer: independent adversarial review (Claude), 2026-09-27.
Diff reviewed: `dafe3b9d2..6a3c5890c` (stacked on #543; 4 commits). Detached worktree
`/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-563` (removed after review), plus a second
worktree at `dafe3b9d2` for the optimizer comparison. R at P1 (`9539352f6`, gllvmTMB 0.7.1,
TMB 1.9.21, R 4.6.0) from the temp library in the builder's scratchpad, used read-only.
Compute: about 25 min wall total.

## Verdict: NON-BLOCKING

No blocking finding. The receipts are R's (bit-for-bit regeneration), the likelihood and
gradient match R at fixed coordinates and both cross-objective directions hold, the sigma_eps
rule is R's rule, the diff stays inside the temporal files, and the tests pass on this machine.
Five non-blocking findings below, two of which I would want fixed before merge (F1, F2).

## Findings

### F1 (non-blocking, fix before merge): the between-optima block is silently skipped on 14 of 25 receipt cells, including `@test jf.converged`

`test/test_temporal_composed_receipts.jl:244` gates the tight comparisons and the convergence
assertion on `interior = jf.hessian_min_eigenvalue > 1e-3 && dll <= 1e-6`. Instrumenting every
cell on this branch (`g_tol = 1e-8`):

```
id,converged,reason,iters,gradnorm,mineig,dll_vs_R_tight,interior_gate
engine__indep_series,false,gradient_not_converged,71,1.34e-07,-6.98e-07,4.08e-09,false
engine__latent_series,false,gradient_not_converged,61,3.44e-07,-3.57e-09,1.37e-09,false
engine__indep_within_unit,false,gradient_not_converged,66,4.68e-07,-9.35e-10,6.22e-12,false
engine__latent_within_unit,false,gradient_not_converged,63,4.94e-08,-1.08e-12,5.00e-11,false
engine__re_int,false,gradient_not_converged,66,1.34e-07,-5.87e-07,3.38e-09,false
sim_u__Tindep_Blatent,true,converged,129,3.10e-12,2.79e-14,3.55e-08,false
sim_u__Tlatentu_Bindep,true,converged,144,2.14e-10,9.62e-14,2.53e-08,false
sim_uo__Tlatentu_Blatent,true,converged,224,1.99e-10,2.35e-14,9.05e-08,false
sim_r__Tindep_Bindep_Windep,true,converged,134,1.19e-12,1.48e-13,5.99e-08,false
... (11 of 25 cells pass the gate)
```

Five engine cells (R's deterministic `value = trait + occasion/10` fixture, degenerate by
construction) report `converged = false` with gradients 5e-8 to 5e-7; four simulated cells have a
flat direction (`mineig ~ 1e-14`) and also skip the block. The likelihood parity on all 25 still
holds (`|Δ logLik|` max 9.0e-8, `dll >= -1e-8` asserted for every cell), so this is not a
correctness gap, but the test cannot tell the reader that only 11 cells are held to the tight
standard, and a regression that pushed more cells out of the gate would pass silently.

Suggested fix: count the interior cells and assert the count (`@test n_interior == 11`), and for
the non-interior cells assert `jf.stopping_reason in (:converged, :gradient_not_converged)` plus a
loose gradient bound (say `<= 1e-6`) so an actual optimizer failure would still fail the file.
State the 11/25 in the PR body's receipt table.

### F2 (non-blocking, fix before merge): the Newton polish ignores the `iterations` budget

`src/temporal_fit.jl:196-233`: after LBFGS stops, the polish loop runs up to 50 Newton steps and
increments `iters`, but never checks `iters < iterations`. Measured on `sim_u__Tindep_Bindep`:

```
iterations=1 -> reported iterations=3 converged=false reason=iteration_limit gradnorm=1.45e+00
iterations=3 -> reported iterations=4 converged=false reason=iteration_limit gradnorm=2.07e+00
```

The reported `iterations` exceeds the caller's cap. The docstring says `iterations = 0` evaluates
at `start` (that path is separate and correct), but any positive cap is soft. Suggested fix:
`for _ in 1:min(50, max(0, iterations - iters))` or break when `iters >= iterations`.

### F3 (non-blocking): the 5e-8 tolerance on the two sigma_eps-fixed cells is narrow and documented, but its justification is Julia-internal

`test_temporal_composed_receipts.jl:174-190` checks the suppressed cells' gradient at R's
optimum against a 256-bit central difference of Julia's own NLL (1e-10) and against R at 5e-8.
The 256-bit reference only proves ForwardDiff is consistent with `temporal_marginal_nll`; it does
not by itself show R's gradient is the noisy one. What does support the claim: (i) NLL and gradient
at the deterministic off-optimum coordinates match R at 1e-8 on those same two cells (max
3.0e-9 / 3.4e-9 over all 25), so the objective is R's; (ii) an R-only check I ran (TMB `gr` vs a
central difference of TMB `fn`, h = 1e-4, at R's `opt$par`):

```
sim_rw__Tindep_Wrowlatent: max|gr - fd(R fn)| = 1.13e-05   (sigma_eps fixed, log_sigma_eps_full = -6.53513)
sim_rw__Tindep_Wrow:       max|gr - fd(R fn)| = 9.60e-06   (sigma_eps fixed)
sim_u__Tindep_Bindep:      max|gr - fd(R fn)| = 4.40e-07   (sigma_eps free)
```

R's own fn/gr are an order of magnitude less self-consistent on the suppressed cells than on a
free-sigma cell, which is directionally consistent with the builder's "inner Laplace at residual
precision ~1e6" explanation. Not a silent widening (in-test comment, decision note, PR body).
Suggested: add the R-side fn-vs-gr discrepancy to the decision note so the justification does not
rest on Julia alone.

### F4 (non-blocking): `update` is a new exported generic named `update`

`src/GLLVModels.jl:376` exports `update`; `src/temporal_methods.jl:824` defines the single method
`update(f::TemporalGaussianFit; ...)`. Checked: `Test.detect_ambiguities(GLLVModels)` is 0,
`methods(update)` is 1, `docs/src/api.md` lists it under Temporal Covariance Source. Semantics
match R's `update.gllvmTMB_multi` (R/methods-gllvmTMB.R:14-37: replay the saved call with named
overrides only) and do not overreach; `start` is not carried over, as in R. The residual concern is
name-space: a short common verb exported from a modelling package will collide with any other
package a user loads that exports `update` (none of GLLVModels' own dependencies does). Consider
either not exporting it (reachable as `GLLVModels.update`) or noting the choice in the decision
note. Not a defect.

### F5 (non-blocking): the spec's "14 blocks (32 expectations)" is a spec typo, not a missing twin

`docs/design/temporal-port-spec.md` (PR #535 head `af130f704`) line 758 says 14 slice-2 blocks / 32
expectations. Its section 4.1 table marks 13 rows "slice 2", summing to 20 expectations
(api.R:44 2, engine.R:34 1, :43 1, :52 2, :69 2, :91 3, :104 2, oracles.R:318 2,
composed-simulation.R:12 1, :36 1, :84 1, bootstrap.R:40 1, selection.R:22 1). The only other row
that touches composition is `sixth-source-engine.R:119` (14 expectations, "T (long form only)"),
which would give 34, not 32, and it is already twinned in slice 1 (`test_temporal_helpers.jl`
testset "temporal latent scores (ar1-methods.R:20, engine.R:119)", 8/8 here). So no block is
missing from this PR; the count in the spec is inconsistent and should be corrected on #535.

## Checks that passed (evidence)

1. **Receipts are R's.** Re-ran `generate_temporal_p1_slice2.R` stage 1 and `cross` in a scratch
   directory against the P1 temp library (stage 1 ~7 min): `diff` against the committed
   `composed.toml` and `composed_cross.toml` is empty (exit 0); sha256
   `bf3cd27d…24b5` and `e5c6e803…feb87` equal `TEMPORAL_P1_SHA256` in `fixture_helpers.jl`.
2. **Cross-objective both ways.** Julia's NLL at R's optimum: `|NLL_J(R par) - R objective|`
   max 3.0e-9 (test, 25 cells, also at R's tight optimum and the off-optimum fixed coordinates).
   R's fn at Julia's optimum: `composed_cross.toml` regenerated by R from the committed
   `julia_optima_composed.csv`, identical; test max 1.7e-9.
3. **Optimizer change on slice 1 fits.** Fitted all 21 `fits.toml` cells at `dafe3b9d2` and at
   `6a3c5890c`: log-likelihoods identical to 1e-12 on every cell (max `dLL` 1.0e-12); six cells
   flipped from `converged = false` (Optim's f-tolerance stop, gradient 3e-8 to 2.6e-7 at
   `g_tol = 1e-8`) to `true` with gradients 1e-12 to 1e-14 after the polish. On three degenerate
   cells (`mineig ~ 0`) the parameter point moved along the flat direction (e.g.
   `forecast__indep__ou` max |Δpar| from R 4.58 → 2.75) with no loglik change. `converged` now
   rests on the recomputed gradient (`maximum(abs, g) <= g_tol`), which is stricter than before,
   so the change does not hide non-convergence. "Keep the lower of two line searches" cannot mask a
   problem because the verdict is taken at the chosen point; cost is roughly double LBFGS work
   (receipt file 138 s here).
4. **Diff scope.** `git diff --stat dafe3b9d2 6a3c5890c`: `src/GLLVModels.jl` (include comments +
   one export), `src/temporal*.jl`, tests, fixtures, docs. No `mixed.jl`, `formula.jl`,
   `Project.toml`; `_laplace_mode` appears only in the after-task prose. The struct
   `TemporalGaussianFit` gained three fields and has a single construction site (updated).
5. **sigma_eps suppression rule.** R (fit-multi.R:6942-6967 at P1): `per_row_diag_W =
   use_diag_W && length(unique(paste(trait_id, site_species_id))) == n_obs` (likewise B), fires
   when `(per_row_diag_W || per_row_diag_B) && !(temporal_active && workflow == "unreplicated")`,
   fixes `log_sigma_eps = log(max(1e-3 sd(y), 1e-6))`. Julia `_temporal_composition`
   (`src/temporal.jl:254-261`) is the same predicate on `(trait_id, level_id)` uniqueness,
   `tier.diag` (indep, or latent with `unique`), `workflow !== :unreplicated`, and `std(y)`.
   R's `use_diag_*` excludes only `.unique_augmented` slopes (`latent(1 + x | …)`), which Julia
   refuses outright. Receipts confirm R suppresses on `latent(d = 1)` per-row unit_obs
   (`sim_rw__Tindep_Wrowlatent`: no `log_sigma_eps` in `opt$par`; `log_sigma_eps_full = -6.53513
   = log(max(1e-3 sd(y), 1e-6))`, verified in R directly). I found no case where one fixes and the
   other does not: a per-row unit (B) diagonal cannot occur with a stable-unit partition and more
   than one occasion, and both sides then agree on the unit_obs case.
6. **Refusal order and limits.** Julia: unit column → unit_obs column → nesting → term grammar →
   series/unit partition (only when a term groups on `unit`, including `(1 | unit)`) → grouping /
   duplicate-per-level. R: nesting (gllvmTMB.R:1180-1186) → parse (1252) → partition (1253-1264,
   `has_stable_unit_component` iterates all `parsed$covstructs`, which include `re_int`) →
   grouping (fit-multi.R:3101). Probed `(1 | unit_group)` with a mismatched series/unit partition:
   refused with R's message; the temporal-only fit on the same data is admitted. One ordinary term
   per level, one `(1 | g)`, `common = true`, and `unique(...)` are refused with `ArgumentError`
   and listed under "Not available yet" in `docs/src/temporal.md`.
7. **Tests.** On this machine (Julia via `julialauncher --project=.`, threads capped):
   `test_temporal_composed_receipts.jl` 643/643, `test_temporal_composed.jl` 55/55,
   `test_temporal_api.jl` 81/81, `test_temporal_oracles.jl` 258/258,
   `test_temporal_fit_receipts.jl` 738/738, `test_temporal_helpers.jl` 129/129 + 8/8. No
   `@test_broken` / `@test_skip` in `test/test_temporal*.jl`. Both new test files carry
   `# gllvm-parity-tag: P1`. The only Julia-against-itself assertions (`frow.sigma_eps ==
   max(1e-3 std(y), 1e-6)` in composed.jl; the oracle's sigma reconstruction in receipts.jl) are
   backed by the R `log_sigma_eps_full` receipt at 1e-12.
8. **`extract_ordination(level = :unit)`** follows R/extractors.R:505-566 (temporal-latent state
   scores when `!use$rr_B`, else conditional `z_B` scores for latent or dep); scores match R at
   8.9e-16 on `engine__latent_B_nounique`.
9. **PR body and docs.** No agent `@handles` in the PR body, after-task, decision note,
   check-log, CHANGELOG or reference page. Parity claims are scoped ("alone or beside ordinary
   unit / unit_obs terms"; cross-source, wide `traits()`, bridge, `gllvm()` hook stated as not
   available); the numbers in the PR table match what I measured. The PR body says "full suite
   not run" plainly.

## What I did not check

- The full `Pkg.test()` suite (Aqua/JET, non-temporal files) — not run here either.
- Julia 1.13; I ran on the default `julialauncher` channel only.
- The Documenter build and `tools/check_reader_surface.py` (claimed in the PR body).
- `simulate` moment agreement beyond the test's own 4 entries; `bootstrap_temporal` /
  `profile_temporal` behaviour on composed fits beyond the refusal path.
- Whether R's `extract_ordination(level = "unit")` on a `dep` unit tier returns what Julia
  returns (the receipt only covers the `latent` cell).
- A Richardson-extrapolated R-side gradient reference for F3 (only the h = 1e-4 check above).
- Performance of `_temporal_series_rows` union-find on large panels.
