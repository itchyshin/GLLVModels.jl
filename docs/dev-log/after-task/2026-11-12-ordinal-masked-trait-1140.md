# After-task: ordinal all-masked trait guard (#1140)

- Branch: `cursor/ordinal-masked-trait-ed18`
- Issue: https://github.com/itchyshin/GLLVModels.jl/issues/1140
- Change: `_check_ordinal_trait_observed` on all three public ordinal fitters;
  regression in `test/test_ordinal_level_check.jl`.
- Checks: targeted `test_ordinal_level_check.jl` on branch (post-commit);
  main `Pkg.test()` started before branch edits (canonical main tally separate).
- Rose: narrow input-validation fix; no API/default change; no parity claim.
