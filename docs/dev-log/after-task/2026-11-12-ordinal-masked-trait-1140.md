# After-task: ordinal all-masked trait guard (#1140)

- Branch: `cursor/ordinal-masked-trait-ed18`
- Issue: https://github.com/itchyshin/GLLVModels.jl/issues/1140
- Change: `_check_ordinal_trait_observed` on all three public ordinal fitters;
  regression in `test/test_ordinal_level_check.jl` (level-below-1, all-masked
  trait row, and `fit_gllvm(...; aghq = :auto)` routes).
- Checks: Julia 1.10.13, 1 thread,
  `julia --project=. --startup-file=no -e 'include("test/test_ordinal_level_check.jl")'`.
  Existing level testset: 36/36 on main and branch. All-masked trait testset on
  main: `fit_ordinal_gllvm` 0/3; `fit_ordinal_gllvm_pertrait` 1/3 (already
  `ArgumentError` from the category guard, not the new trait guard);
  `fit_ordinal_gllvm_pertrait_cov` 1/3 (same). On branch: 3/3 per fitter, 9/9.
  AGHQ public route: 1/1 on branch. Full `Pkg.test` not run.
- Rose: narrow input-validation fix; no API/default change; no parity claim.
