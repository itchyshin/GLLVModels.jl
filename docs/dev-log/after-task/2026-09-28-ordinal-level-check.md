# After-task: ordinal fitters reject observed levels below 1 (2026-09-28)

## 1. Goal

Stop an out-of-bounds memory write in the ordinal fitters. `_pack_initial_ordinal_pertrait` in `src/families/ordinal.jl` counts categories with `counts[Int(Y[t, i])] += 1` inside an `@inbounds` loop, and the shared-cutpoint fitter has the same pattern. No fitter checked that observed levels are at least 1, so a level of 0 or a negative level indexed the count vector out of bounds. Goal: every public ordinal fitter throws a clear `ArgumentError` before any level-indexed `@inbounds` loop, masked cells stay allowed, valid data is unchanged.

## 2. Implemented

- `_check_ordinal_levels(Y, obs)` in `src/families/ordinal.jl`: walks the observed cells and throws `ArgumentError` naming the offending value and its index, with the text "levels must be integers 1..C". Masked cells (`obs` false) are skipped.
- Called immediately after `obs` is built in the three public entry points: `fit_ordinal_gllvm`, `fit_ordinal_gllvm_pertrait`, `fit_ordinal_gllvm_pertrait_cov`. This precedes the category-count loop, the warm starts and `_pack_initial_ordinal_pertrait`.
- `test/test_ordinal_level_check.jl` (36 assertions), registered in `test/runtests.jl` after `test_ordinal_logit_twin.jl`.
- CHANGELOG bullet under Development.

## 3. Decisions and Rejected Alternatives

- One check at each entry point instead of a guard inside `_pack_initial_ordinal_pertrait`. The shared-cutpoint fitter does not call that helper but has its own `counts[Int(Ys[i])]` loop, so a guard in the helper alone would miss it.
- Only a lower bound is checked. Levels above the current maximum set `C`, so they cannot overflow; that behaviour is left alone as instructed.
- No integer check beyond the signature: all three fitters already require `AbstractMatrix{<:Integer}`, so non-integer levels cannot arrive.
- Rejected: removing `@inbounds`. That changes performance on valid data and would still leave a raw `BoundsError` instead of a clear message.

## 4. Files Touched

- `src/families/ordinal.jl`
- `test/test_ordinal_level_check.jl` (new)
- `test/runtests.jl` (one include line)
- `CHANGELOG.md`
- `docs/dev-log/check-log.md`
- `docs/dev-log/after-task/2026-09-28-ordinal-level-check.md` (this file)

## 5. Checks Run

All single-file runs, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`.

- RED on origin/main `85b7a688d` (scratch detached worktree, since removed), Julia 1.10.12: 12 pass, 12 fail, 12 error of 36, both default bounds mode and `--check-bounds=yes`.
- GREEN on the branch: 36/36 on Julia 1.10.12, 36/36 on 1.10.12 with `--check-bounds=yes`, 36/36 on 1.13.0.
- Neighbours on 1.10.12, each file alone: bridge_missing_mask 92/92, bridge_x 200/200, confint_family 341/341, core070_link_boundaries 21/21, diagnostics 65/65, extractors 92/92, lv_ci 196/196, missing_data 34/34, ordinal_fit 10/10, ordinal_link_input 49/49, ordinal_logit_twin 29/29, ordinal_pertrait 113/113, ordinal_probit 10/10, ordinal_x_identity 21/21, postfit 1106/1106, second_order_ordinal_pertrait_ci 26 pass plus 1 environment-gated `@test_skip`, statsapi 74/74.

## 6. Tests of the Tests

- The RED run shows the test detects the bug on main: for each of the three fitters, the level-0 and level-(-1) cases fail (4 failures and 4 errors per fitter), while the masked-placeholder and valid-data cases pass (4 passes per fitter). So the masked and valid assertions do not depend on the fix, and the invalid-level assertions do.
- On main the shared route throws `BoundsError` in both bounds modes; the per-trait routes return without error, which is what `@test_throws` caught.
- The message assertions check that the offending value and "1..C" appear, so a generic `ArgumentError` from elsewhere would not pass.

## 7. Issue Ledger

- Fixed: out-of-bounds write for observed levels below 1 in the three ordinal fitters.
- Not an issue, noted: `src/phylo_ordinal_xlv.jl` has a similar `counts[yi] += 1` loop under `@inbounds`, but it already range-checks `1 <= yi <= C` before indexing. No change.

## 8. Consistency Audit

- Searched `src/` for `counts[` and for every caller of the three fitters (`bridge.jl`, `formula.jl`, `fit_gllvm.jl`, `confint_family.jl` bootstrap refits). All reach the fitters through the public entry points, so all now get the check. Bootstrap refits should simulate levels in `1:C` and so not trip the check (inference, not tested).
- The mask convention (`obs = mask === nothing ? trues(p, n) : mask`) is unchanged; the helper reads the same `obs`.

## 9. What Did Not Go Smoothly

- `test/test_missing_data.jl` errors when run alone (`UndefVarError: Poisson`) because it relies on `Distributions` being loaded by an earlier file in the full suite. Rerunning with `using Distributions` first gave 34/34, so the error comes from running the file alone and is unrelated to the fix.
- The first RED run output was noisy because `err.msg` errors on a `BoundsError`; counts were taken from the test summary.

## 10. Known Residuals

- `eachindex(Y, obs)` throws `DimensionMismatch` if a mask has a different shape from `Y`. Before, a wrong-shaped mask would have indexed out of bounds or silently misaligned, so only invalid input behaves differently. (Inference: no caller passes a mismatched mask; not measured beyond the neighbour runs.)
- The full `Pkg.test()` suite was not run; only the listed files.
- Not pushed; no PR.

## 11. Team Learning

- `@inbounds` around a loop indexed by user data is a validation debt: the check must sit at the entry point, before the first such loop, and the RED test should run in both bounds modes because the failure looks different in each.

## 12. Cross-Product Coverage

- Shared cutpoints, per-trait cutpoints, per-trait with covariates: each covered for level 0, level -1, a masked invalid placeholder, and valid data.
- Logit link only in the new test; the check runs before any link-specific code, so probit takes the same path (inference, not separately tested).
- `fit_gllvm(Y; family = Ordinal())`, the bridge and `@formula` routes reach the fix through `fit_ordinal_gllvm_pertrait` / `_cov`; not tested directly with invalid levels.
