# A0 first output (preliminary): carry-rule scan at contract level (2026-09-27)

Method: every tracked file on GLLVModels.jl origin/main (1385b0490) under `docs/` that has a `source_pins` block (53 files) was parsed. Pins naming gllvmTMB sources (`R/*.R`, `src/gllvmTMB.cpp`, including readback paths under `.unlazy/.../oracle-source/readback/` and `gllvmTMB/R/...`) were checked against `git diff --name-only b4d5fee64 9539352f6 -- R src` in the gllvmTMB clone (74 files: 62 changed, 12 added).

Result:
- 29 files pin gllvmTMB sources. **All 29 are stale under the carry rule (D-295 row 1)**: each pins at least one file that changed, usually `R/fit-multi.R` or `R/gllvmTMB.R`, which almost every contract depends on. Worst: namespace-1 (19 of 22 pins changed), postfit-surface-inventory (27 of 44), postfit-2 (10 of 21).
- 12 evidence files (aghq-*, family-boundary-*, family-link-*, package-qualification, masks-known-contract) pin only Julia-side files and parity fixture scripts (`test/parity/...R`), no gllvmTMB source. Under the carry rule they cannot show byte-identical R sources, so they do not carry automatically either; WS0 must add the R pins or re-measure.
- 12 files have an empty or absent R pin list.

Reading: at contract level nothing carries. Expect the row-level count (WS0, via the ledger's row-to-receipt binding) to be close to "every R-bound row is PARTIAL_STALE_AT_P1". The re-pin is therefore mostly re-measurement at P1, not bookkeeping, and A3's compute (kohaku or DRAC while Totoro is down) is on the critical path.

Not covered: row-level mapping (which ledger rows cite which contract), and whether a changed file changed in a way that matters to the row (the rule is byte-identity, so this does not affect the count, only what a re-measurement is likely to find).
